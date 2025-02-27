import jax
import jax.numpy as jnp                # JAX NumPy
from jax import nn as jnn              # JAX nn
import argparse

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import time
from absl import app
from functools import partial
from absl import flags
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
#import tensorflow_datasets as tfds     # TFDS for MNIST
from keras.datasets import mnist
from typing import (Any, List)

import enum
import functools
import itertools
from typing import (Any, Callable, cast, List, NamedTuple, Optional, Tuple,
                    TypeVar, Union)

from absl import logging
import chex
from flax import struct
import jax
from jax import lax
from jax.experimental import pjit
from jax.experimental.sparse import linalg
import jax.numpy as jnp
import numpy as np
import optax

# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper routines for quantization."""

from typing import Any

import chex
from flax import struct
import jax.numpy as jnp


# pylint:disable=no-value-for-parameter
@struct.dataclass
class QuantizedValue:
  """State associated with quantized value."""
  quantized: chex.Array
  diagonal: chex.Array  # Diagonal (if extract_diagonal is set)
  bucket_size: chex.Array
  quantized_dtype: jnp.dtype = struct.field(
      pytree_node=False)  # Dtype for the quantized value.
  extract_diagonal: bool = struct.field(
      pytree_node=False)  # In case its centered.
  shape: Any = struct.field(pytree_node=False)  # Shape of the tensor.

  @classmethod
  def from_float_value(cls, fvalue, quantized_dtype, extract_diagonal=False):
    if isinstance(fvalue, list) and not fvalue:
      return QuantizedValue([], [], [], quantized_dtype, extract_diagonal, [])
    quantized, diagonal_fvalue, bucket_size = QuantizedValue.quantize(
        fvalue, quantized_dtype, extract_diagonal)
    return QuantizedValue(quantized, diagonal_fvalue, bucket_size,
                          quantized_dtype, extract_diagonal,
                          list(quantized.shape))

  # Quantization is from Lingvo JAX optimizers.
  # We extend it for int16 quantization of PSD matrices.
  @classmethod
  def quantize(cls, fvalue, quantized_dtype, extract_diagonal=False):
    """Returns quantized value and the bucket."""
    if quantized_dtype == jnp.float32:
      return fvalue, [], []
    elif quantized_dtype == jnp.bfloat16:
      return fvalue.astype(jnp.bfloat16), [], []

    float_dtype = fvalue.dtype
    if quantized_dtype == jnp.int8:
      # value -128 is not used.
      num_buckets = jnp.array(127.0, dtype=float_dtype)
    elif quantized_dtype == jnp.int16:
      # value -32768 is not used.
      num_buckets = jnp.array(32767.0, dtype=float_dtype)
    else:
      raise ValueError(f'Quantized dtype {quantized_dtype} not supported.')
    # max value is mapped to num_buckets

    if extract_diagonal and fvalue.ndim != 2:
      raise ValueError(
          f'Input array {fvalue} must be 2D to work with extract_diagonal.')

    diagonal_fvalue = []
    if extract_diagonal:
      diagonal_fvalue = jnp.diag(fvalue)
      # Remove the diagonal entries.
      fvalue = fvalue - jnp.diag(diagonal_fvalue)

    # TODO(rohananil): Extend this by making use of information about the blocks
    # SM3 style which will be useful for diagonal statistics
    # We first decide the scale.
    if fvalue.ndim < 1:
      raise ValueError(
          f'Input array {fvalue} must have a strictly positive number of '
          'dimensions.')

    max_abs = jnp.max(jnp.abs(fvalue), axis=0)
    bucket_size = max_abs / num_buckets
    bs_expanded = bucket_size[jnp.newaxis, Ellipsis]
    # To avoid divide by 0.0
    bs_nonzero = jnp.where(bs_expanded > 0.0, bs_expanded,
                           jnp.ones_like(bs_expanded))
    ratio = fvalue / bs_nonzero
    # We use rounding to remove bias.
    quantized = jnp.round(ratio)
    return quantized.astype(quantized_dtype), diagonal_fvalue, bucket_size

  def to_float(self):
    """Returns the float value."""
    if isinstance(self.quantized, list) and not self.quantized:
      return self.quantized

    if self.quantized_dtype == jnp.float32:
      return self.quantized.astype(jnp.bfloat16)

    if self.quantized_dtype == jnp.bfloat16:
      return self.quantized.astype(jnp.bfloat16)

    float_dtype = self.bucket_size.dtype
    bucket_size = self.bucket_size[jnp.newaxis, Ellipsis]
    val = self.quantized.astype(float_dtype) * bucket_size
    if self.extract_diagonal:
      val += jnp.diag(self.diagonal)
    return val


# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX Ops for symmetric matrices used by the Shampoo optimizer."""

import functools
from typing import Any, List, Optional, Sequence, Union

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class SlicedSymmetricMatrix:
  """A symmetric matrix represented by lower-triangular block row slices.
  For example, the symmetric matrix M = [[a, b^T], [b, c]] would be represented
  by the block rows a and [b, c].
  The matrix may be batched, in which case each entry of block_rows may have
  dimension greater than 2. The last two dimensions represent the rows and cols.
  """
  block_rows: List[jnp.ndarray]


def product_with_transpose(
    mat1,
    mat2,
    axes,
    precision = lax.Precision.DEFAULT,
):
  """Returns mat1 * mat2^T for two matrices (possibly batched).
  The rows and columns are the last two dimensions for each matrix.
  Args:
    mat1: First matrix.
    mat2: Second matrix.
    axes: The axes over which to apply the product.
    precision: JAX precision to use for the multiplication.
  """
  return jnp.tensordot(a=mat1, b=mat2, axes=axes, precision=precision)


@functools.partial(jax.jit, static_argnames=("block_size", "axes", "precision"))
def sliced_transposed_product(
    mat,
    block_size,
    axes = (-1,),
    precision = lax.Precision.DEFAULT,
):
  """Returns the blocked slices representing a symmetric contraction.
  Specifically, the output is a contraction of the input mat with itself, in the
  specified axes.
  Args:
    mat: The matrix for which we will compute a contraction with itself.
    block_size: The size of row blocks to compute.
    axes: Axes to use for the contraction.
    precision: The precision to use in each computation.
  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  rank = len(mat.shape)

  def _make_axis_positive(ax):
    assert -rank <= ax < rank
    return ax + rank if ax < 0 else ax

  positive_axes = [_make_axis_positive(ax) for ax in axes]
  assert len(positive_axes) == len(axes)
  remaining_axes = set(range(rank)) - set(positive_axes)
  assert len(remaining_axes) == 1
  remaining_ax = remaining_axes.pop()

  num_rows = mat.shape[remaining_ax]
  if num_rows % block_size != 0:
    raise ValueError(
        "The row dimension must be divisible by block_size. "
        f"Instead got row dimension={num_rows} and block_size={block_size}.")

  block_rows = []
  for i in range(num_rows // block_size):
    start_indices = [0]*rank
    start_indices[remaining_ax] = i * block_size

    slice_sizes = list(mat.shape)
    slice_sizes[remaining_ax] = block_size

    slice_sizes_full = list(mat.shape)
    slice_sizes_full[remaining_ax] = (i + 1) * block_size

    block_rows.append(
        product_with_transpose(
            lax.dynamic_slice(
                mat, start_indices=start_indices, slice_sizes=slice_sizes),
            lax.dynamic_slice(
                mat, start_indices=[0] * rank, slice_sizes=slice_sizes_full),
            axes=(axes, axes),
            precision=precision))

  return SlicedSymmetricMatrix(block_rows=block_rows)


@functools.partial(jax.jit, static_argnames=("block_size", "axes", "precision"))
def sliced_transposed_product_concat(
    mat,
    block_size,
    axes = (-1,),
    precision = lax.Precision.DEFAULT,
):
  """Returns the concatenated slices representing mat*mat^T.
  Args:
    mat: The matrix for which we will compute mat*mat^T. It does not need to be
      square, and may be batched.
    block_size: The size of row blocks to compute.
    axes: Axes to use for the contraction.
    precision: The precision to use in each computation.
  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  sliced_symmetric_matrix = sliced_transposed_product(
      mat=mat, block_size=block_size, axes=axes, precision=precision)
  return jnp.concatenate(sliced_symmetric_matrix.block_rows, axis=-1)


@jax.jit
def materialize_matrix(symmetric_matrix):
  """Returns a materialized symmetric matrix.
  Args:
    symmetric_matrix: the matrix represented by lower-triangular block slices.
  """
  block_rows = symmetric_matrix.block_rows
  block_size = block_rows[0].shape[-2]
  num_blocks = len(block_rows)

  # Slice the lower-triangular and diagonal blocks into blocks.
  blocks = [[
      block_row[Ellipsis, i * block_size:(i + 1) * block_size] for i in range(k + 1)
  ] for k, block_row in enumerate(block_rows)]

  # Generate the (off-diagonal) upper-triangular blocks.
  off_diags = [[] for _ in range(num_blocks - 1)]
  for k, block_row in enumerate(block_rows[1:]):
    for i in range(k + 1):
      off_diags[i].append(
          jnp.swapaxes(
              a=block_row[Ellipsis, i * block_size:(i + 1) * block_size],
              axis1=-1,
              axis2=-2))

  return jnp.block([row + row_t for row, row_t in zip(blocks[:-1], off_diags)] +
                   [blocks[-1]])


@functools.partial(jax.jit, static_argnames=("num_blocks"))
def materialize_matrix_from_concat(
    block_rows_concat,
    num_blocks = None,
):
  """Returns a materialized symmetric matrix from concatenated slices.
  Args:
    block_rows_concat: The matrix represented as the concatenated
      lower-triangular blocks.
    num_blocks: The number of block-rows used to represent the symmetric matrix.
      If not specified, it is inferred from the shape of block_rows_concat.
  """
  if num_blocks is None:
    num_blocks = find_num_blocks(block_rows_concat)

  block_size = block_rows_concat.shape[-2]

  block_rows = [
      block_rows_concat[Ellipsis, (k * (k + 1)) // 2 *
                        block_size:(((k + 1) * (k + 2)) // 2 + 1) * block_size]
      for k in range(num_blocks)
  ]

  return materialize_matrix(SlicedSymmetricMatrix(block_rows=block_rows))


@functools.partial(jax.jit, static_argnames=("alpha", "beta", "axes"))
def update_sliced_rows(
    symmetric_matrix,
    mat,
    alpha,
    beta,
    axes = (-1,),
):
  """Implements the blocked equivalent of SYRK.
  Specifically, the symmetric matrix (represented using lower-triangular block
  rows) is updated using the sliced product of mat.
  Args:
    symmetric_matrix: The symmetric matrix to update.
    mat: The matrix to use for the update = mat * mat^T. The number of rows
      should match that of symmetric_matrix.
    alpha: The weight for the update.
    beta: The weight for the original symmetric matrix.
    axes: Axes to use for the contraction of the update.
  Returns:
    The updated rows of alpha * mat * mat^T + beta * symmetric_matrix.
  """
  block_size = symmetric_matrix.block_rows[0].shape[-2]
  sym_prod = sliced_transposed_product(
      mat=mat, block_size=block_size, axes=axes)
  return SlicedSymmetricMatrix(block_rows=[
      update * alpha + row * beta
      for update, row in zip(sym_prod.block_rows, symmetric_matrix.block_rows)
  ])


def num_blocks_from_total_blocks(total_blocks):
  """Returns the number of blocks (i.e.
  block rows) from the total blocks.
  This is the inverse of the function x -> x*(x+1)/2.
  For example, the matrix M = [[A, B^T], [B, C]] may be represented using a
  total of 3 blocks ([A, B, C]). The number of corresponding block rows is 2.
  Args:
    total_blocks: The total blocks used to represent the matrix.
  """
  num_blocks = np.round(
      (np.sqrt(8 * total_blocks + 1) - 1) / 2).astype(np.int32)
  if (num_blocks * (num_blocks + 1)) / 2 != total_blocks:
    raise ValueError(
        f"total_blocks={total_blocks} does not correspond to "
        "a symmetric matrix. It must have the form total_blocks = x*(x+1)/2.")
  return num_blocks


def find_num_blocks(block_rows_concat):
  """Returns the number of (row) blocks representing the concatenated matrix.
  For example, an input with dimensions [256, 2560] represents 10 square blocks,
  which matches 4 lower-triangular block rows (1+2+3+4). So this function will
  return 4.
  Use ordinary numpy functions here so that the returned value is static.
  Args:
    block_rows_concat: The concatenated block array.
  Raises:
    ValueError: When the dimensions of the matrix do not correspond to a lower
    triangular block representation.
  """
  # Compute the number of square blocks used to represent the matrix.
  total_blocks = block_rows_concat.shape[-1] / block_rows_concat.shape[-2]
  # Determine the number of block rows by inverting y = x*(x+1)/2.
  return num_blocks_from_total_blocks(total_blocks)


@functools.partial(jax.jit, static_argnames=("block_size"))
def slice_symmetric_matrix(
    mat,
    block_size,
):
  """Returns sliced row blocks.
  Args:
    mat: A symmetric matrix.
    block_size: The size of the row slices.
  """
  num_rows = mat.shape[-2]
  num_cols = mat.shape[-1]
  if num_rows != num_cols:
    raise ValueError("mat is not square.")
  if num_rows % block_size != 0:
    raise ValueError("block size does not evenly divide rows. "
                     f"num_rows={num_rows}, block_size={block_size}")
  return SlicedSymmetricMatrix(block_rows=[
      mat[Ellipsis, i * block_size:(i + 1) * block_size, 0:(i + 1) * block_size]
      for i in range(num_rows // block_size)
  ])


@functools.partial(jax.jit, static_argnames=("block_size"))
def slice_symmetric_matrix_concat(
    mat,
    block_size,
):
  """Returns the concatenated sliced row blocks.
  Args:
    mat: A symmetric matrix.
    block_size: The size of the row slices.
  """
  sliced_symmetric_matrix = slice_symmetric_matrix(
      mat=mat, block_size=block_size)
  return jnp.concatenate(sliced_symmetric_matrix.block_rows, axis=-1)


def sliced_matrix_diag(mat):
  """Returns the diagonal of the symmetric matrix.
  Args:
    mat: The symmetric matrix represented in concatenated block form.
  """
  rows, cols = mat.shape
  total_blocks = cols // rows
  num_blocks = num_blocks_from_total_blocks(total_blocks)
  diags = []
  for i in range(num_blocks):
    last_index = rows * ((i+2) * (i+1)) // 2
    first_index = last_index - rows
    diags.append(jnp.diag(mat[Ellipsis, first_index:last_index]))
  return jnp.concatenate(diags, axis=-1)


def diag_as_concat(diag, block_size):
  """Returns the representation of a diagonal matrix in symmetric block form.
  Args:
    diag: The 1D array for the diagonals.
    block_size: The size of blocks to use. Must divide the length of diag.
  """
  assert len(diag.shape) == 1  # diag must be 1D.
  assert len(diag) % block_size == 0
  num_diag_blocks = len(diag) // block_size
  blocks = []
  for i in range(num_diag_blocks):
    blocks.append(
        jnp.zeros(shape=(block_size, block_size * i), dtype=diag.dtype))
    blocks.append(jnp.diag(diag[i * block_size:(i + 1) * block_size]))
  return jnp.concatenate(blocks, axis=-1)


def row_abs_maxes(mat):
  """Returns the max of the absolute values of the rows of the full matrix.
  For example the symmetric matrix M = [[1, 6], [6, 2]] is represented using
  mat = [1, 6, 2] with block_size = 1. In this case the function returns the
  aboslute row maxes of the original symmetric matrix, [6, 6].
  Args:
    mat: The symmetric matrix represented as the concatenated blocks.
  """
  rows, cols = mat.shape

  # Find col and row max for each block.
  col_maxes = []
  row_maxes = []
  for i in range(cols // rows):
    block = jnp.abs(mat[Ellipsis, i * rows:(i + 1) * rows])
    col_maxes.append(jnp.max(block, axis=1))
    row_maxes.append(jnp.max(block, axis=0))

  # global row max from block maxes.
  num_blocks = num_blocks_from_total_blocks(cols // rows)
  maxes = []
  for i in range(num_blocks):
    maxes.append(
        jnp.concatenate(
            row_maxes[(i * (i + 1) // 2):((i + 2) * (i + 1) // 2)] + [
                col_maxes[((j + 1) * (j + 2)) // 2 - (j - i + 1)]
                for j in range(i + 1, num_blocks)
            ],
            axis=-1))

  return jnp.max(jnp.stack(maxes), axis=0)


def times_vector(mat, vec):
  """Returns the symmetric block-concatenated matrix multiplied by a vector.
  Specifically, each value in the vector is multiplied by a row of the full
  matrix. That is, the vector is broadcast and multiplied element-wise. Note
  this would be the transpose of full_mat * vec if full_mat represented the full
  symmetric matrix.
  Args:
    mat: The symmetric matrix represented as the concatenated blocks.
    vec: The vector, having the same dimension as the materialized matrix.
  """
  rows, cols = mat.shape
  num_blocks = num_blocks_from_total_blocks(cols // rows)
  multiplied = []
  for i in range(num_blocks):
    mat_block = mat[Ellipsis,
                    rows * ((i + 1) * i) // 2:rows * ((i + 1) * (i + 2)) // 2]
    vec_block = vec[Ellipsis, rows * i:rows * (i + 1)]
    multiplied.append(jnp.einsum("...ij,...i->ij", mat_block, vec_block))
  return jnp.concatenate(multiplied, axis=-1)


# Dtype for inverse-pth root routine
# Switch to f64 if you have hardware that supports it. Enable the jax flag
# jax_enable_x64 for this to work, otherwise it will default to float32.
_MAT_INV_PTH_ROOT_DTYPE = jnp.bfloat16


def _default_zero_field():
  return struct.field(
      default_factory=functools.partial(jnp.array, 0, jnp.bfloat16))


InversePthRootDiagnosticsSubtype = TypeVar(
    "InversePthRootDiagnosticsSubtype", bound="InversePthRootDiagnostics")


@struct.dataclass
class InversePthRootDiagnostics:
  """Diagnostics for inverse p-th root iterative procedure.
  Given an inverse pth root B = A^(-1/p), contains the average and
  maximum diagonal and off diagonal absolute entrywise errors between
  (B^p A) and I.
  """
  max_diag_error: chex.Array = _default_zero_field()
  avg_diag_error: chex.Array = _default_zero_field()
  max_off_diag_error: chex.Array = _default_zero_field()
  avg_off_diag_error: chex.Array = _default_zero_field()
  p: chex.Array = _default_zero_field()

  @classmethod
  def create(cls,
             pth_inverse_root, matrix,
             p):
    """Generates a diagnostics struct from (-1/p) root result."""
    mat_m = jnp.matmul(
        mat_power(pth_inverse_root, p),
        matrix,
        precision=jax.lax.Precision.DEFAULT)
    num_off_diag_entries = mat_m.size - jnp.diag(mat_m).size
    diag_error = jnp.abs(jnp.diag(mat_m) - 1).astype(jnp.bfloat16)
    off_diag_error = jnp.abs(mat_m - jnp.diag(jnp.diag(mat_m))).astype(
        jnp.bfloat16)
    return cls(
        max_diag_error=jnp.max(diag_error).astype(jnp.bfloat16),
        avg_diag_error=jnp.mean(diag_error).astype(jnp.bfloat16),
        max_off_diag_error=jnp.max(off_diag_error).astype(jnp.bfloat16),
        avg_off_diag_error=(jnp.sum(off_diag_error) /
                            num_off_diag_entries).astype(jnp.bfloat16),
        p=jnp.array(p, jnp.bfloat16))


LOBPCGDiagnosticsSubtype = TypeVar(
    "LOBPCGDiagnosticsSubtype", bound="LOBPCGDiagnostics")


@struct.dataclass
class LOBPCGDiagnostics:
  """Diagnostics for iterative LOBPCG eigenvalue routine.
  Contains consistency error for LOBPCG eigenvalue routine, which
  refers to |A v - lambda v| / (lambda + |A v|) for a proposed eigenpair
  (v, lambda). This metics dataclass retains consistency error
  and other useful LOBPCG values.
  """
  lobpcg_iters: chex.Array = _default_zero_field()
  max_consistency_error: chex.Array = _default_zero_field()
  avg_consistency_error: chex.Array = _default_zero_field()
  # Average of absolute value of off-diagonal of V^T V for eigenvalues V.
  avg_orthogonality_error: chex.Array = _default_zero_field()
  max_eigenvalue: chex.Array = _default_zero_field()
  min_eigenvalue: chex.Array = _default_zero_field()
  num_topk_eigenvectors: chex.Array = _default_zero_field()

  @classmethod
  def create(cls, matrix,
             eigvals, eigvecs,
             lobpcg_iters):
    """Generates LOBPCG diagnostics from the result of the routine."""
    num_topk = len(eigvals)
    num_off_diag = num_topk * (num_topk - 1)
    precision = jax.lax.Precision.DEFAULT

    mat_eigvecs = matrix.dot(eigvecs, precision=precision)
    consistency_error_unnormalized = jnp.linalg.norm(
        mat_eigvecs - eigvals * eigvecs, ord=2, axis=0)
    normalization = jnp.linalg.norm(mat_eigvecs, ord=2, axis=0) + eigvals
    consistency_error = consistency_error_unnormalized / normalization

    orthogonality_error = eigvecs.T.dot(eigvecs, precision=precision)
    orthogonality_error -= jnp.diag(jnp.diag(orthogonality_error))

    return cls(
        lobpcg_iters=jnp.array(lobpcg_iters, jnp.bfloat16),
        max_consistency_error=jnp.max(consistency_error).astype(jnp.bfloat16),
        avg_consistency_error=jnp.mean(consistency_error).astype(jnp.bfloat16),
        avg_orthogonality_error=(jnp.sum(orthogonality_error) /
                                 num_off_diag).astype(jnp.bfloat16),
        max_eigenvalue=jnp.max(eigvals).astype(jnp.bfloat16),
        min_eigenvalue=jnp.min(eigvals).astype(jnp.bfloat16),
        num_topk_eigenvectors=jnp.array(num_topk, jnp.bfloat16),
    )


@struct.dataclass
class TrainingMetrics:
  """Diagnostic metrics from training."""
  # Error for inverse-pth roots.
  inverse_pth_root_errors: chex.Array = _default_zero_field()
  # Iteration count for inverse-pth roots.
  inverse_pth_root_iters: chex.Array = _default_zero_field()
  # If final iteration error increases sufficiently, iteration terminates early.
  # This field records the ratio of the final iteration error.
  final_error_ratio: chex.Array = _default_zero_field()
  lobpcg_diagnostics: LOBPCGDiagnostics = struct.field(
      default_factory=LOBPCGDiagnostics)
  # Rich matrix entrywise error diagnostics, if enabled.
  inverse_pth_root_diagnostics: InversePthRootDiagnostics = struct.field(
      default_factory=InversePthRootDiagnostics)
  # Diagnostics applied to the conditioned p-th root problem, after top
  # eigenvectors are removed, if LOBPCG is being applied.
  conditioned_inverse_pth_root_diagnostics: InversePthRootDiagnostics = (
      struct.field(default_factory=InversePthRootDiagnostics))
  # TODO(rohananil): Add more important metrics to track during training.


# Per parameter optimizer state used in data-parallel training.
class ParameterStats(NamedTuple):
  """State associated to each parameter of the model being trained."""
  diagonal_statistics: QuantizedValue  # Accumulator for diagonal preconditioner
  statistics: List[Any]  # Statistics (QuantizedValue, chex.Array)
  preconditioners: List[Any]  # Preconditioners (QuantizedValue, chex.Array)
  diagonal_momentum: QuantizedValue  # Momentum for the diagonal preconditioner
  momentum: QuantizedValue  # Momentum for the shampoo preconditioner
  training_metrics: Union[TrainingMetrics, optax.MaskedNode]  # Optional.


# For training extremely large model; We keep a global state with a concatenated
# statistics and preconditioner states for all vars. This is so that we can
# annotate the leading axis to be sharded to save memory at the cost of
# communication.
@struct.dataclass
class GlobalShardedParameterStats:
  statistics: chex.Array  # Statistics
  preconditioners: chex.Array  # Preconditioners
  exponents: chex.Array  # exponents


# These are per-parameter local states; All statistics here mirror the parameter
# Thus the sharding is copied over from the param specification.
@struct.dataclass
class LocalShardedParameterStats:
  """State associated to each parameter of the model being trained."""
  diagonal_statistics: QuantizedValue  # Accumulator for diagonal preconditioner
  diagonal_momentum: QuantizedValue  # Momentum for the diagonal preconditioner
  momentum: QuantizedValue  # Momentum for the shampoo preconditioner
  training_metrics: Union[TrainingMetrics, optax.MaskedNode]
  index_start: np.int32 = struct.field(
      pytree_node=False)  # Index into global statistics array
  sizes: Any = struct.field(pytree_node=False)  # Sizes of the statistics.


def init_training_metrics(
    num_statistics, generate_training_metrics
):
  if not generate_training_metrics:
    return optax.MaskedNode()
  return jax.tree_map(
      functools.partial(jnp.repeat, repeats=num_statistics), TrainingMetrics())


def init_training_metrics_shapes(
    num_statistics, generate_training_metrics
):
  seed = init_training_metrics(num_statistics, generate_training_metrics)
  return jax.tree_map(lambda arr: [list(arr.shape), arr.dtype], seed)


def init_training_metrics_pspec(
    generate_training_metrics
):
  if not generate_training_metrics:
    return optax.MaskedNode()
  return jax.tree_map(lambda _: pjit.PartitionSpec(), TrainingMetrics())


class ShardedShampooStats(NamedTuple):
  """Shampoo state in sharded mode."""
  global_stats: Any
  local_stats: Any


class ShampooState(NamedTuple):
  count: chex.Array
  stats: Any


class InitFnState(NamedTuple):
  init_fn: Any
  pspec_fn: Any
  shape_and_dtype_fn: Any


class GraftingType(enum.IntEnum):
  SGD = 1
  ADAGRAD = 2
  RMSPROP = 3
  RMSPROP_NORMALIZED = 4
  SQRT_N = 5
  ADAGRAD_NORMALIZED = 6


class PreconditionerType(enum.IntEnum):
  # Default, computes preconditioner for each dim
  ALL = 1
  # One sided Shampoo, in this cases only on input dim.
  # Assumes last dim is always the output dim and everything else input dim.
  INPUT = 2


def power_iteration(
    matrix,
    num_iters = 100,
    error_tolerance = 1e-6,
    precision = lax.Precision.DEFAULT,
    padding_start = None,
):
  r"""Power iteration algorithm.
  The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
  a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
  of `A`, and a vector v, which is the corresponding eigenvector of `A`.
  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)
  Args:
    matrix: the symmetric PSD matrix.
    num_iters: Number of iterations.
    error_tolerance: Iterative exit condition.
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise) b)
      lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
      (best possible precision, slowest)
    padding_start: if set, assumes rows and columns after padding_start are
      zero.
  Returns:
    eigen vector, eigen value
  """
  matrix_size = matrix.shape[-1]

  def _iter_condition(state):
    i, unused_v, unused_s, unused_s_v, run_step = state
    return jnp.logical_and(i < num_iters, run_step)

  def _iter_body(state):
    """One step of power iteration."""
    i, new_v, s, s_v, unused_run_step = state
    new_v = new_v / jnp.linalg.norm(new_v)

    s_v = jnp.einsum("ij,j->i", matrix, new_v, precision=precision)
    s_new = jnp.einsum("i,i->", new_v, s_v, precision=precision)
    return (i + 1, s_v, s_new, s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance))

  # Figure out how to use step as seed for random.
  v_0 = np.random.RandomState(1729).uniform(-1.0, 1.0,
                                            matrix_size).astype(matrix.dtype)
  v_0 = jnp.array(v_0)
  if padding_start is not None:
    v_0 *= (jnp.arange(len(v_0), dtype=jnp.int32) < padding_start)

  init_state = tuple([0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True])
  _, v_out, s_out, _, _ = lax.while_loop(_iter_condition, _iter_body,
                                         init_state)
  v_out = v_out / jnp.linalg.norm(v_out)
  return v_out, s_out


def mat_power(
    mat_m,
    p,
    precision = lax.Precision.DEFAULT,
):
  """A simple matrix power method. M^p where p can be TracedValue."""
  power = jnp.eye(mat_m.shape[0], dtype=_MAT_INV_PTH_ROOT_DTYPE)

  def _iter_condition(state):
    i, _, _ = state
    return i > 0

  def _iter_body(state):
    i, power, mat = state

    power = jax.lax.cond(i % 2 == 1,
                         lambda: jnp.matmul(mat, power, precision=precision),
                         lambda: power)
    i //= 2
    mat = jnp.matmul(mat, mat, precision=precision)
    return i, power, mat

  _, result, _ = lax.while_loop(_iter_condition, _iter_body, (p, power, mat_m))
  return result


def _pth_root_difference(w, alpha, beta,
                         p):
  """Computes (w+alpha)^(-1/p)-(w+beta)^(-1/p)."""

  a = w + alpha
  b = w + beta
  a_minus_b = alpha - beta
  exp = -1 / p

  def _stable_subtract(b, a_minus_b):
    # Mathematically identical to the target expression, with (w+beta)^(-1/p)
    # term factored out and w cancellation in the subtraction.
    return (b**exp) * jnp.expm1(exp * jnp.log1p(a_minus_b / b))

  return jnp.where(
      # Choose the branch with the best log1p approximation.
      jnp.abs(a_minus_b / b) < jnp.abs(a_minus_b / a),
      -_stable_subtract(a, -a_minus_b),
      _stable_subtract(b, a_minus_b))


def matrix_inverse_pth_root(
    matrix,
    p,
    num_iters = 100,
    ridge_epsilon = 1e-6,
    error_tolerance = 1e-6,
    precision = lax.Precision.DEFAULT,
    relative_matrix_epsilon = True,
    lobpcg_topk_precondition = 0,
    lobpcg_max_iter = 0,
    padding_start = None,
):
  """Computes `matrix^(-1/p)`, where `p` is a positive integer.
  This function uses the Coupled newton iterations algorithm for
  the computation of a matrix's inverse pth root.
  References:
    [Functions of Matrices, Theory and Computation,
     Nicholas J Higham, Pg 184, Eq 7.18](
     https://epubs.siam.org/doi/book/10.1137/1.9780898717778)
  Args:
    matrix: the symmetric PSD matrix whose power it to be computed
    p: exponent, for p a positive integer.
    num_iters: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
    error_tolerance: Error indicator, useful for early termination.
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise) b)
      lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
      (best possible precision, slowest)
    relative_matrix_epsilon: Whether to use relative epsilon to the max eigen
      value when computing inverse-pth root.
    lobpcg_topk_precondition: If nonzero, specifies the number of top
      eigenvectors to subtract out before performing LOBPCG. Note this makes
      relative_matrix_epsilon essentially free.
    lobpcg_max_iter: Maximum iteration count for LOBPCG, defaults to
      `lobpcg_topk_precondition`.
    padding_start: If the input matrix was padded, then zeros out columns
      and rows at the padding start.
  Returns:
    `(matrix + eps)^(-1/p)` and error metrics.
    Note `eps` is not added to zeroed out padding rows and
    columns. `eps` is just `ridge_epsilon` if
    `relative_matrix_epsilon` is set to `False`, otherwise, it is the
    ridge epsilon value scaled by the derived maximum eigenvalue of
    the input matrix.
  """

  # If the input is not square, materialize it from the concatenated form.
  if matrix.shape[0] != matrix.shape[1]:
    matrix = materialize_matrix_from_concat(matrix)

  assert matrix.shape[0] == matrix.shape[1]

  # We use _MAT_INV_PTH_ROOT_DTYPE for the matrix inverse pth root.
  # Switch to f64 if you have hardware that supports it. Enable the jax flag
  # jax_enable_x64 for this to work.
  matrix_size = matrix.shape[0]
  orig_dtype = matrix.dtype
  matrix = matrix.astype(_MAT_INV_PTH_ROOT_DTYPE)
  alpha = jnp.asarray(-1.0 / p, _MAT_INV_PTH_ROOT_DTYPE)
  identity = jnp.eye(matrix_size, dtype=_MAT_INV_PTH_ROOT_DTYPE)

  if padding_start is not None:
    # Zero out padding in identity as well for convergence checks.
    ix = (jnp.arange(matrix_size, dtype=jnp.int32) < padding_start).astype(
        matrix.dtype)
    matrix *= ix[jnp.newaxis, :]
    matrix *= ix[:, jnp.newaxis]
    identity *= ix

  original_matrix = matrix

  # Only used in lobpcg branches, but required by pytype.
  eigvals, eigvecs, lobpcg_diagnostics = None, None, None
  if lobpcg_topk_precondition > 0:
    # TODO(vladf): reuse previous top-k as the initial search directions
    pad_shape = (matrix_size - lobpcg_topk_precondition,
                 lobpcg_topk_precondition)
    search_dirs = jnp.concatenate(
        (jnp.eye(lobpcg_topk_precondition), jnp.zeros(pad_shape)), axis=0)
    eigvals, eigvecs, lobpcg_iters = linalg.lobpcg_standard(
        matrix, search_dirs,
        lobpcg_topk_precondition if lobpcg_max_iter == 0 else lobpcg_max_iter)
    lobpcg_diagnostics = LOBPCGDiagnostics.create(
        matrix,
        eigvals,
        eigvecs,
        lobpcg_iters,
    )

    # The minimal eigenvalue among top-k becomes the maximal one in the whole
    # matrix after deflation.
    deflation = eigvals - jnp.min(eigvals)
    scaled_vecs = eigvecs * jnp.sqrt(deflation)

    # Deflate out top eigenvectors to reduce matrix condition number.
    matrix -= scaled_vecs.dot(
        scaled_vecs.T, precision=jax.lax.Precision.DEFAULT)

  if relative_matrix_epsilon:
    if eigvals is not None:
      max_ev = jnp.max(eigvals)
    else:
      # Only use power iteration if lobpcg wasn't already used to derive the
      # top eigenvalue.
      _, max_ev = power_iteration(
          matrix=matrix,
          num_iters=100,
          error_tolerance=1e-6,
          precision=precision,
          padding_start=padding_start)
  else:
    # Use absolute matrix epsilon scaling otherwise.
    max_ev = 1.0

  ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-6)
  # Sometimes error increases after an iteration before decreasing and
  # converging. 1.2 factor is used to bound the maximal allowed increase.
  max_error_ratio = 1.2

  def _iter_condition(state):
    i, unused_mat_m, unused_mat_h, unused_old_mat_h, error, error_ratio = state
    error_above_threshold = jnp.logical_and(error > error_tolerance,
                                            error_ratio < max_error_ratio)
    return jnp.logical_and(i < num_iters, error_above_threshold)

  def _iter_body(state):
    (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=precision)
    new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=precision)
    new_error = jnp.max(jnp.abs(new_mat_m - identity))
    return (i + 1, new_mat_m, new_mat_h, mat_h, new_error, new_error / error)

  if matrix_size == 1:
    damped_matrix = matrix + ridge_epsilon
    resultant_mat_h = damped_matrix**alpha
    error = jnp.array(0, jnp.bfloat16)
    iters = 0
    error_ratio = 0.0
  else:
    damped_matrix = matrix + ridge_epsilon * identity

    z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
    new_mat_m_0 = damped_matrix * z
    new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
    iters, mat_m, mat_h, old_mat_h, error, error_ratio = lax.while_loop(
        _iter_condition, _iter_body, init_state)
    error = jnp.max(jnp.abs(mat_m - identity)).astype(jnp.bfloat16)
    is_converged = jnp.asarray(error_ratio < max_error_ratio, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h

  conditioned_resultant_mat = resultant_mat_h

  if lobpcg_topk_precondition > 0:
    # Since we deflated the top eigenvectors prior to p-th root inverse,
    # the resultant matrix has larger eigenvalues associated with those
    # same eigenvectors, which we need to now re-deflate.
    #
    # Note that _pth_root_difference returns positive values for this
    # particular argument ordering as min(eigvals) <= eigvals for the
    # jnp.sqrt below.
    pth_diff = _pth_root_difference(ridge_epsilon, jnp.min(eigvals), eigvals, p)
    scaled_vecs = eigvecs * jnp.sqrt(pth_diff)
    resultant_mat_h = conditioned_resultant_mat - scaled_vecs.dot(
        scaled_vecs.T, precision=jax.lax.Precision.DEFAULT)

  error_metrics = TrainingMetrics(
      inverse_pth_root_errors=jnp.array(error, jnp.bfloat16),
      inverse_pth_root_iters=jnp.array(iters, jnp.bfloat16),
      final_error_ratio=jnp.array(error_ratio, jnp.bfloat16))

  if lobpcg_topk_precondition > 0:
    conditioned_diagnostics = InversePthRootDiagnostics.create(
        conditioned_resultant_mat, damped_matrix, p)
    unconditioned_damped_matrix = original_matrix + ridge_epsilon * identity
    unconditioned_diagnostics = InversePthRootDiagnostics.create(
        resultant_mat_h, unconditioned_damped_matrix, p)
    # The max entrywise error in error_metrics.inverse_pth_root_errors refers
    # to what was derived from the inverse pth root iteration, which with
    # LOBPCG refers to the conditioned problem. Make sure to use the error
    # from the unconditioned problem.
    unconditional_errors = jnp.maximum(
        unconditioned_diagnostics.max_diag_error,
        unconditioned_diagnostics.max_off_diag_error)
    error_metrics = error_metrics.replace(
        inverse_pth_root_errors=unconditional_errors,
        lobpcg_diagnostics=lobpcg_diagnostics,
        conditioned_inverse_pth_root_diagnostics=conditioned_diagnostics,
        inverse_pth_root_diagnostics=unconditioned_diagnostics,
    )

  if padding_start is not None:
    # Occasionally, pure-padding matrices are handed to the inversion routine
    # due to some TPU hosts not having the same number of preconditioning
    # matrices.
    resultant_mat_h = jnp.where(padding_start == 0, 0.0, resultant_mat_h)
    error = jnp.where(padding_start == 0, 0.0,
                      error_metrics.inverse_pth_root_errors)
    error_metrics = error_metrics.replace(inverse_pth_root_errors=error)

  resultant_mat_h = jnp.asarray(resultant_mat_h, orig_dtype)
  return resultant_mat_h, error_metrics




def merge_small_dims(shape_to_merge, max_dim):
  """Merge small dimensions.
  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]
  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.
  Returns:
    Merged shape.
  """
  if shape_to_merge and np.all(np.array(shape_to_merge) == 1):
    return [1]

  resulting_shape = []
  product = 1
  for d in shape_to_merge:
    if product * d <= max_dim:
      product *= d
    else:
      if product > 1:
        resulting_shape.append(product)
      product = d
  if product > 1:
    resulting_shape.append(product)
  return resulting_shape


def pad_square_matrix(mat, max_size):
  """Pad a square matrix up to max_size.
  Args:
    mat: a matrix to pad.
    max_size: matrix size requested.
  Returns:
    Given M returns [[M, 0], [0, I]]
  """
  rows, cols = mat.shape
  if rows != cols:
    raise ValueError("Must have rows == cols, instead got "
                     f"rows={rows}, cols={cols}")
  if cols > max_size:
    raise ValueError("Must have cols <= max_size. Instead got "
                     f"cols={cols}, max_size={max_size}.")
  if rows == max_size:
    return mat
  pad_size = max_size - rows

  zs1 = jnp.zeros([rows, pad_size], dtype=mat.dtype)
  zs2 = jnp.zeros([pad_size, rows], dtype=mat.dtype)
  eye = jnp.eye(pad_size, dtype=mat.dtype)
  mat = jnp.concatenate([mat, zs1], 1)
  mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
  return mat


def make_sliced_padding(
    symmetric_block_size,
    num_blocks,
    starting_block,
    dtype,
):
  """Returns padding for symmetric block matrix.
  Specifically, the padding is given concatenated rectangular matrices
  representing the lower-triangular rows below the starting block. For example,
  if we want to pad the symmetric matrix
  M = [[A, B^T]
       [B, C]],
  the desired output (in terms of the full matrix) with num_blocks = 4 is
  M_padded = [[A, B^T, 0, 0]
              [B, C,   0, 0]
              [0, 0,   I, 0]
               0, 0,   0, I].
  We would represent M as the block matrix mat = [A, B, C]. In this form, the
  additional padding to provide has form [0, 0, I, 0, 0, 0, I] (only the lower
  triangular parts in the third and fourth rows).
  Args:
    symmetric_block_size: The size of each block.
    num_blocks: The total number of blocks.
    starting_block: The block where to start the padding.
    dtype: The type to use for the blocks.
  """
  if starting_block == num_blocks:
    return jnp.zeros(shape=(symmetric_block_size, 0), dtype=dtype)

  blocks = []
  for i in range(starting_block, num_blocks):
    blocks.append(
        jnp.zeros(
            shape=(symmetric_block_size, symmetric_block_size * i),
            dtype=dtype))
    blocks.append(jnp.eye(symmetric_block_size, dtype=dtype))
  return jnp.concatenate(blocks, axis=-1)


def pad_block_symmetric_matrix(
    mat,
    symmetric_block_size,
    max_num_blocks,
):
  """Returns the padded blocked symmetric matrix.
  The size of the padded matrix will be:
    [symmetric_block_size, symmetric_block_size * max_num_blocks]
  The input matrix can either:
    - Be square with size less or equal to symmetric_block_size. In this case,
      mat will first be padded to a square matrix of size symmetric_block_size,
      and then be padded again up to the full size of the blocked matrix.
    - Be a rectangle with number of rows equal to block size.
      In this case, number of columns must be a multiple of number of rows, and
      the ratio must correspond to a block representation of a symmetric matrix.
      That is, the ratio must have form x * (x + 1) / 2. Here, x represents the
      number of block rows represented by the matrix.
  Args:
    mat: The input block matrix.
    symmetric_block_size: The size of blocks.
    max_num_blocks: The largest number of blocks to pad to.
  """
  rows, cols = mat.shape
  if rows > symmetric_block_size:
    raise ValueError(
        "Must have rows <= symmetric_block_size. Instead got "
        f"rows={rows}, symmetric_block_size={symmetric_block_size}.")
  if rows > cols:
    raise ValueError("Must have rows <= cols, instead got "
                     f"rows={rows}, cols={cols}.")
  if cols > symmetric_block_size * max_num_blocks:
    raise ValueError("Must have cols <= symmetric_block_size * max_num_blocks "
                     f"Instead got cols={cols}, "
                     f"symmetric_block_size={symmetric_block_size}, "
                     f"max_num_blocks={max_num_blocks}.")
  if rows < symmetric_block_size:
    mat = pad_square_matrix(mat, max_size=symmetric_block_size)
  # Update rows and cols after possibly padding in pad_square_matrix.
  rows, cols = mat.shape
  assert rows == symmetric_block_size
  assert cols % rows == 0
  filled_blocks = cols // rows
  padding_blocks = make_sliced_padding(
      symmetric_block_size=symmetric_block_size,
      num_blocks=num_blocks_from_total_blocks(
          max_num_blocks),
      starting_block=num_blocks_from_total_blocks(
          filled_blocks),
      dtype=mat.dtype)
  return jnp.concatenate([mat, padding_blocks], axis=-1)


def pad_vector(vec, max_size):
  """Pad a vector to a max_size.
  Args:
    vec: a vector to pad.
    max_size: matrix size requested.
  Returns:
    Given V returns [V, 0]
  """
  size = vec.shape[0]
  assert size <= max_size
  if size == max_size:
    return vec
  pad_size = max_size - size
  zs1 = jnp.zeros([pad_size], dtype=vec.dtype)
  return jnp.concatenate([vec, zs1], 0)


def efficient_cond(predicate, compute_fn, init_state, *args, **kwargs):
  """Avoids wasteful buffer allocation with XLA."""

  def _iter_body(unused_state):
    results = compute_fn(*args, **kwargs)
    return tuple([False] + list(results))

  def _iter_condition(state):
    return state[0]

  results = jax.lax.while_loop(_iter_condition, _iter_body,
                               tuple([predicate] + init_state))
  return tuple(results[1:])


class BlockPartitioner:
  """Partitions a tensor into smaller tensors."""

  def __init__(self, param, block_size):
    self._shape = param.shape
    self._splits = []
    split_sizes = []
    # We split params into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(param.shape):
      if 0 < block_size < d:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d - 1) // block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        split_sizes.append(sizes)
      else:
        split_sizes.append(np.array([d], dtype=np.int32))
    self._split_sizes = split_sizes

  def split_sizes(self):
    return self._split_sizes

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, indices) in self._splits:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
      tensors = tensors_local
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            jnp.concatenate(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


def gram_weighted_update(
    old_stats,
    g,
    axis,
    w1,
    w2,
    precision = None):
  """Updated statistics via weighted average with new Gram matrix.
    Returns w₁ R + w₂ Gᵀ G where R is `old_stats` and G is the matrix whose
    columns are the flattened slices of the tensor `g` along the given `axis`.
    (So, `old_stats` and the returned matrix have dimensions n x n where
    n = `g.shape[axis]`).
  Args:
    old_stats:  Old statistics.
    g:  Gradient tensor.
    axis:  Axis along which to slice `g`.
    w1:  Scalar weight for old statistics.
    w2:  Scalar weight for new Gram matrix.
    precision: Optional precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise)
      b) lax.Precision.HIGH (increased precision, slower)
      c) lax.Precision.HIGHEST (best possible precision, slowest)
  Returns:
    Weighted average of old and new statistics.
  """
  axes = [i for i in range(g.ndim) if i != axis]
  gram_matrix = jnp.tensordot(g, g, axes=(axes, axes), precision=precision)
  return w1 * old_stats + w2 * gram_matrix




class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self,
               param,
               block_size,
               merge_small_dims_block_size,
               best_effort_shape_interpretation,
               preconditioner_type=PreconditionerType.ALL):
    """Initializes the preconditioner.
    Args:
      param: parameter to precondition.
      block_size: Block size used to split param.
      merge_small_dims_block_size: Block size for merging dims.
      best_effort_shape_interpretation: Whether to collapse/merge dims together.
      preconditioner_type: Type of preconditioner to use.
    """
    self._original_shape = param.shape
    self._transformed_shape = param.shape
    if best_effort_shape_interpretation:
      self._transformed_shape = merge_small_dims(
          self._original_shape, merge_small_dims_block_size)
    reshaped_param = jnp.reshape(param, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_param, block_size)
    self._preconditioner_type = preconditioner_type

  def updated_statistics_from_grad(
      self,
      stats,
      grad,
      w1,
      w2,
      to_float = None,
      from_float = None,
      precision = None,
  ):
    """Update statistics from gradients.
    Args:
      stats: Old statistics or its Cholesky factor if `cholesky` is True.
      grad: Gradient to compute statistics from.
      w1: Weight for old statistics.
      w2: Weight for new statistics.
      to_float: Optional function for converting stats to floating point.
      from_float: Optional function for converting from floating point.
      precision: Optional precision XLA related flag, the available options are:
        a) lax.Precision.DEFAULT (better step time, but not precise)
        b) lax.Precision.HIGH (increased precision, slower)
        c) lax.Precision.HIGHEST (best possible precision, slowest)
    Returns:
      A list of updated gradient statistics for each partition.
    """
    to_float = to_float if to_float is not None else (lambda x: x)
    from_float = from_float if from_float is not None else (lambda x: x)
    update = functools.partial(gram_weighted_update, precision=precision)
    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    new_stats = []
    index = 0
    for g in partitioned_grads:
      should_preconditioned_dims = self.should_precondition_dims()
      num_preconditioners = sum(should_preconditioned_dims)
      for axis in range(num_preconditioners):
        new_stat = update(to_float(stats[index]), g, axis, w1, w2)
        new_stats.append(from_float(new_stat))
        index += 1
    return new_stats

  def should_precondition_dims(self):
    """A vector containing indicator indicating if the dim is preconditioned."""
    split_sizes = self._partitioner.split_sizes()
    rank = len(split_sizes)
    if self._preconditioner_type == PreconditionerType.ALL or rank <= 1:
      return [True] * rank
    else:
      return [True] * (rank - 1) + [False]

  def shapes_for_preconditioners(self):
    """Returns shape from statistics."""
    split_sizes = self._partitioner.split_sizes()
    rank = len(split_sizes)
    # We ignore preconditioner types if rank == 1
    preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      if self._preconditioner_type == PreconditionerType.ALL or rank <= 1:
        preconditioner_shapes.extend([[d, d] for d in t])
      else:
        preconditioner_shapes.extend([[d, d] for d in t[:-1]])
    return preconditioner_shapes

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    should_preconditioned_dims = self.should_precondition_dims()
    num_preconditioners = sum(should_preconditioned_dims)
    return 2 * num_preconditioners

  def preconditioned_grad(self, grad, preconditioners):
    """Precondition the gradient.
    Args:
      grad: A gradient tensor to precondition.
      preconditioners: A list of preconditioners to apply.
    Returns:
      A preconditioned gradient.
    """

    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    for i, g in enumerate(partitioned_grads):
      should_preconditioned_dims = self.should_precondition_dims()
      num_preconditioners = sum(should_preconditioned_dims)
      preconditioners_for_grad = preconditioners[i *
                                                 num_preconditioners:(i + 1) *
                                                 num_preconditioners]
      precond_g = g
      rank = len(g.shape)
      for j, precondition in enumerate(should_preconditioned_dims):
        if precondition:
          precond_g = jnp.tensordot(
              precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
        else:
          precond_g = jnp.transpose(precond_g, axes=(*range(1, rank), 0))
      preconditioned_partitioned_grads.append(precond_g)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return jnp.reshape(merged_grad, self._original_shape)


def _convert_to_parameter_stats(global_stats,
                                local_stat,
                                convert_statistics=True):
  """Creates parameter stats from sharded stats."""
  index_start = int(local_stat.index_start)
  index_end = int(len(local_stat.sizes)) + index_start
  statistics = global_stats.statistics[index_start:index_end, :, :]
  preconditioners = global_stats.preconditioners[index_start:index_end, :, :]
  new_statistics = []
  new_preconditioners = []
  for i, size in enumerate(local_stat.sizes):
    new_statistics.append(statistics[i][:size, :size])
    new_preconditioners.append(preconditioners[i][:size, :size])
  if not convert_statistics:
    new_statistics = None
  return ParameterStats(local_stat.diagonal_statistics, new_statistics,
                        new_preconditioners, local_stat.diagonal_momentum,
                        local_stat.momentum, local_stat.training_metrics)


def _convert_from_parameter_stats(parameter_stats, local_stats):
  """Creates sharded stats from paramter stats."""
  return LocalShardedParameterStats(parameter_stats.diagonal_statistics,
                                    parameter_stats.diagonal_momentum,
                                    parameter_stats.momentum,
                                    parameter_stats.training_metrics,
                                    local_stats.index_start, local_stats.sizes)


def _add_metrics_into_local_stats(local_stats, metrics, keep_old):
  """Adds errors back into local statistics."""
  new_local_stats = []
  for local_stat in local_stats:
    index_start = int(local_stat.index_start)
    index_end = int(len(local_stat.sizes)) + index_start
    # pylint:disable=cell-var-from-loop Used immediately.
    per_stat_metrics = jax.tree_map(lambda x: x[index_start:index_end], metrics)
    # We don't want to update the metrics if we didn't do a new inverse p-th
    # root calculation to find a new preconditioner, so that TensorBoard curves
    # look consistent (otherwise they'd oscillate between NaN and measured
    # values).
    per_stat_metrics = efficient_cond(keep_old,
                                      lambda: [local_stat.training_metrics],
                                      [per_stat_metrics])[0]
    # pylint:enable=cell-var-from-loop
    new_local_stats.append(
        local_stat.replace(training_metrics=per_stat_metrics))
  return new_local_stats


def batch(x, num_devices):
  """Batch `x` so that so that leading axis is num_devices."""
  n = len(x)
  b = int(n / num_devices)
  return jnp.stack([jnp.stack(x[idx:idx + b]) for idx in range(0, n, b)])


def unbatch(batched_values):
  """Unbatch values across leading axis and return a list of elements."""
  b1, b2 = batched_values.shape[0], batched_values.shape[1]
  results = []
  for v_array in jnp.split(batched_values, indices_or_sections=b1, axis=0):
    v_array = jnp.squeeze(v_array)
    # b2 = batches (number of preconditioner computation) per core.
    if b2 > 1:
      for v in jnp.split(v_array, indices_or_sections=b2, axis=0):
        results.append(jnp.squeeze(v))
    else:
      results.append(v_array)
  return results


def distributed_shampoo(
    learning_rate,
    block_size,
    beta1=0.9,
    beta2=0.999,
    diagonal_epsilon=1e-10,
    matrix_epsilon=1e-6,
    weight_decay=0.0,
    start_preconditioning_step=5,
    preconditioning_compute_steps=1,
    statistics_compute_steps=1,
    best_effort_shape_interpretation=True,
    graft_type=GraftingType.SGD,
    nesterov=True,
    exponent_override=0,
    # Pass pmap 'batch axis name' in pmap mode.
    batch_axis_name=None,
    ### Only set following 3 params in pjit/spmd mode.
    ### WARNING: Experimental
    statistics_partition_spec=None,
    preconditioner_partition_spec=None,
    num_devices_for_pjit=None,
    shard_optimizer_states=False,
    ###
    ### Experimental memory reduction mode
    best_effort_memory_usage_reduction=False,
    ###
    inverse_failure_threshold=0.1,
    moving_average_for_momentum=False,
    skip_preconditioning_dim_size_gt=4096,
    clip_by_scaled_gradient_norm=None,
    precision=lax.Precision.DEFAULT,
    tensordot_precision = None,
    relative_matrix_epsilon=True,
    merge_small_dims_block_size=4096,
    lobpcg_topk_precondition = 0,
    lobpcg_max_iter = 0,
    precondtioner_type=PreconditionerType.ALL,
    skip_preconditioning_rank_lt=1,
    decoupled_learning_rate=True,
    decoupled_weight_decay=False,
    generate_training_metrics=True,
):
  """Distributed Shampoo optimizer.
  Distributed Shampoo is a second-order preconditioned method (concretely, a
  variant of full-matrix Adagrad), that provides significant convergence and
  wall-clock time improvements compared to conventional first-order methods,
  and that has been shown to scale to large state-of-the-art deep learning
  models.
  References:
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    Preprint: https://arxiv.org/abs/2002.09018
  Args:
    learning_rate: the step size used to update the parameters.
    block_size: Block size for large layers (if > 0). Preconditioning compute
      operation is cubic in the dimension of the tensor. Block size allows us to
      chunk the layers into sub-layers of maximal dimension dictated by this
      value. Use 128 as default (increase if you have compute budget).
    beta1: momentum parameter.
    beta2: second moment averaging parameter.
    diagonal_epsilon: epsilon for diagonal adagrad (only if layerwise grafting
      to AdaGrad is enabled).
    matrix_epsilon: epsilon to add to statistics before computing inverse pth
      root. If you are running in f32 precision for inverse pth root
      (recommended today) this can go upto 1e-6. If you have latest hardware
      with native f64 precision, set this upto 1e-12.
    weight_decay: Weight decay for regularization.
    start_preconditioning_step: When to start Shampoo update before which
      diagonal update is used. This is because we dont have enough information
      to do stable inverse.
    preconditioning_compute_steps: How often to compute preconditioner.
      Performance tuning params for controlling memory and compute requirements.
      Ideally set this and statistics_compute_steps params to 1.
    statistics_compute_steps: How often to compute statistics.
    best_effort_shape_interpretation: If there are some small dimensions,
      collapse them e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if
      block = 1024, [1, 2, 768, 1, 2048] --> [2, 768, 2048]
    graft_type: Grafting is a technique to fix the layerwise scale of Shampoo
      optimizer. This allows us to plugin the Shampoo optimizer into settings
      where SGD/AdaGrad is already well tuned.
    nesterov: Nesterov momentum.
    exponent_override: Override the exponent used in matrix inverse.
    batch_axis_name: labeled axis over pmap for data-parallel training the
      optimizer used for.
    statistics_partition_spec: PartitionSpec to be used in sharded mode.
    preconditioner_partition_spec: PartitionSpec to be used in sharded mode.
    num_devices_for_pjit: Number of devices to parallelize over when using pjit.
    shard_optimizer_states: Shard optimizer states to save memory in model
      parallel training.
    best_effort_memory_usage_reduction: Best effort memory usage reduction. -
      diagonal_statistics -> jnp.bfloat16 - momentum buffers (2x) -> jnp.int8 -
      statistics, preconditioners -> jnp.int16 + diagonals
    inverse_failure_threshold: numerics are hard and inverses fail sometimes; we
      determine that using this threshold.
    moving_average_for_momentum: Whether to use moving average for momentum
      instead of exponential moving average.
    skip_preconditioning_dim_size_gt: Skip if preconditioning dim size is
      greater than this value.
    clip_by_scaled_gradient_norm: Clip by scaled gradient norm (only useful when
      using RMSProp Grafting).
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise) b)
      lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
      (best possible precision, slowest)
    tensordot_precision: Optional precision to use for the tensordot operation
      when computing statistics (e.g., G Gᵀ). Same options as `precision` above.
    relative_matrix_epsilon: Whether to use relative epsilon to the max eigen
      value when computing inverse-pth root.
    merge_small_dims_block_size: Used as the maximum block size
      to merge the shapes.
    lobpcg_topk_precondition: If nonzero, specifies the number of top
      eigenvectors to subtract out before performing LOBPCG. Note this makes
      relative_matrix_epsilon essentially free.
    lobpcg_max_iter: Number of LOBPCG iterations, if zero defaults to
      `lobpcg_topk_precondition`.
    precondtioner_type: Preconditioner type to select all, left only or right
      only preconditioners.
    skip_preconditioning_rank_lt: Skips preconditioning for parameters with
      rank less than this value.
    decoupled_learning_rate: If True, use decoupled learning rate, otherwise
      couple it with preconditioned gradient computation. (Default True)
    decoupled_weight_decay: If True, use decoupled weight decay, otherwise
      couple with weight decay. (Default False)
    generate_training_metrics: If True, gather training metrics, otherwise
      avoid generating them (to reduce memory usage).
  Returns:
    a GradientTransformation.
  """

  def _graft_type_has_diagonal_statistics():
    """Returns True if using diagonal firt order method for grafting."""
    return graft_type != GraftingType.SGD and graft_type != GraftingType.SQRT_N

  def quantized_dtype_for_momentum_buffers(var):
    return jnp.int8 if best_effort_memory_usage_reduction and len(
        var.shape) > 1 else jnp.bfloat16

  # Preconditioner and statistics are both stores as int16 in this mode.
  # We take out the diagonal to make quantization easier.
  def quantized_dtype_for_second_moment_statistics_buffers():
    return jnp.int16 if best_effort_memory_usage_reduction and batch_axis_name else jnp.bfloat16

  # Preconditioner and statistics are both stores as int16 in this mode.
  # We take out the diagonal to make quantization easier.
  def quantized_dtype_for_second_moment_preconditioner_buffers():
    return jnp.int16 if best_effort_memory_usage_reduction and batch_axis_name else jnp.bfloat16

  def _to_float(maybe_quantized):
    if isinstance(maybe_quantized, QuantizedValue):
      return maybe_quantized.to_float()
    else:
      return maybe_quantized

  def _maybe_quantize_statistics(statistics_list):
    return _maybe_quantize_matrices_with_dtype(
        statistics_list, quantized_dtype_for_second_moment_statistics_buffers())

  def _maybe_quantize_preconditioners(statistics_list):
    return _maybe_quantize_matrices_with_dtype(
        statistics_list,
        quantized_dtype_for_second_moment_preconditioner_buffers())

  def _maybe_quantize_matrices_with_dtype(statistics_list, quantized_dtype):
    if quantized_dtype != jnp.bfloat16:
      return ([
          QuantizedValue.from_float_value(
              s, quantized_dtype, extract_diagonal=True)
          for s in statistics_list
      ])
    else:
      return statistics_list

  def _maybe_dequantize_preconditioners(preconditioner_list):
    return _maybe_dequantize_matrices_with_dtype(
        preconditioner_list,
        quantized_dtype_for_second_moment_preconditioner_buffers())

  def _maybe_dequantize_matrices_with_dtype(statistics_list, quantized_dtype):
    if quantized_dtype != jnp.bfloat16:
      return [s.to_float() for s in statistics_list]
    else:
      return statistics_list

  def _quantize_diagonal_statistics(diagonal_statistics):
    return QuantizedValue.from_float_value(diagonal_statistics, jnp.bfloat16)

  def _quantize_momentum(momentum_statistics):
    return QuantizedValue.from_float_value(
        momentum_statistics,
        quantized_dtype_for_momentum_buffers(momentum_statistics))

  def preconditioner_from_params(param):
    """Returns a Preconditioner object for given param."""
    return Preconditioner(param, block_size, merge_small_dims_block_size,
                          best_effort_shape_interpretation, precondtioner_type)

  def sharded_init_fn(params):
    """Returns optimizer state (for PJIT mode).
    Args:
      params: the parameters that should be updated.
    """
    params_flat, treedef = jax.tree_flatten(params)
    # Find max size to pad to.
    max_size = 0
    for param in params_flat:
      preconditioner = preconditioner_from_params(param)
      if not _skip_preconditioning(param):
        shapes = preconditioner.shapes_for_preconditioners()
        sizes = [s[0] for s in shapes]
        max_size = max(max(sizes), max_size)

    padded_statistics = []
    padded_preconditioners = []
    local_stats_flat = []
    exponents = []
    for param in params_flat:
      preconditioner = preconditioner_from_params(param)
      shapes = preconditioner.shapes_for_preconditioners()
      sizes = []

      statistics = []
      preconditioners = []
      index_start = len(padded_statistics)
      if not _skip_preconditioning(param):
        sizes = [s[0] for s in shapes]
        shapes = preconditioner.shapes_for_preconditioners()
        statistics = [
            matrix_epsilon * jnp.eye(max_size, dtype=jnp.bfloat16)
            for s in shapes
        ]
        preconditioners = [jnp.eye(max_size, dtype=jnp.bfloat16) for s in shapes]
        padded_statistics.extend(statistics)
        padded_preconditioners.extend(preconditioners)
        exponent = (
            preconditioner.exponent_for_preconditioner()
            if exponent_override == 0 else exponent_override)
        exponents.extend([exponent] * len(shapes))

      diagonal_statistics = _quantize_diagonal_statistics(
          jnp.zeros_like(param))
      diagonal_momentum = _quantize_momentum(jnp.zeros_like(param))
      momentum = _quantize_momentum(jnp.zeros_like(param))

      local_stats_flat.append(
          LocalShardedParameterStats(
              diagonal_statistics, diagonal_momentum, momentum,
              init_training_metrics(len(sizes), generate_training_metrics),
              index_start, sizes))

    local_stats = jax.tree_unflatten(treedef, local_stats_flat)
    to_pad = -len(padded_statistics) % num_devices_for_pjit
    if max_size == 0:
      to_pad = num_devices_for_pjit
      max_size = block_size
      stat_dtype = jnp.bfloat16
    else:
      stat_dtype = padded_statistics[0].dtype
    # Pad the statistics and preconditioner matrices to be a multiple of
    # num devices.
    # TODO(rohananil): Relax to only the size of the mesh axis where the dim
    # is split on.
    padded_statistics.extend(
        [jnp.eye(max_size, dtype=stat_dtype) for _ in range(to_pad)])
    padded_preconditioners.extend(
        [jnp.eye(max_size, dtype=stat_dtype) for _ in range(to_pad)])
    exponents.extend([1 for _ in range(to_pad)])
    global_stats = GlobalShardedParameterStats(
        jnp.stack(padded_statistics), jnp.stack(padded_preconditioners),
        jnp.stack(exponents))
    return ShampooState(
        count=jnp.zeros([], jnp.int32),
        stats=ShardedShampooStats(global_stats, local_stats))


  def _max_statistics_size_from_params(params):
    max_size = 0
    for param in params:
      param_clone = jnp.zeros(param.shape, dtype=param.dtype)
      preconditioner = preconditioner_from_params(param_clone)
      if not _skip_preconditioning(param):
        shapes = preconditioner.shapes_for_preconditioners()
        sizes = [s[0] for s in shapes]
        max_size = max(max(sizes), max_size)
    return max_size

  def _remove_leading_sharding_annotation(pspec):
    """Mapping from N-d to (N-1)-d, used for quantization, factoring etc."""
    # None and PSpec(None) are valid PSpecs.
    if pspec and len(pspec) > 1:
      return pjit.PartitionSpec(*pspec[1:])
    else:
      return []

  def sharded_init_partition_spec_fn(params, params_partition_spec,
                                     partition_spec_for_statistics):
    """Returns a parallel state tree with PartitionSpec associated with state.
    Args:
      params: A pytree with params.
      params_partition_spec: A pytree with PartitionSpec for params.
      partition_spec_for_statistics: PartitionSpec for the statistics.
    """
    # Parallel lists of spec, and params.
    param_pspec_flat, _ = jax.tree_flatten(
        params_partition_spec, is_leaf=lambda x: x is None)
    params_flat, treedef = jax.tree_flatten(params)
    assert param_pspec_flat
    assert params_flat
    # Step is replicated across cores.
    # None means cores.
    local_stats_flat = []
    num_statistics = 0
    for param, param_pspec in zip(params_flat, param_pspec_flat):
      param_clone = jnp.zeros(param.shape, dtype=param.dtype)
      preconditioner = preconditioner_from_params(param_clone)
      shapes = preconditioner.shapes_for_preconditioners()
      sizes = []

      index_start = num_statistics
      if not _skip_preconditioning(param):
        sizes = [s[0] for s in shapes]
        shapes = preconditioner.shapes_for_preconditioners()
        num_statistics += len(shapes)

      qdtype = quantized_dtype_for_momentum_buffers(param)
      m1_pspec = param_pspec
      m2_pspec = param_pspec
      m1_scale_pspec = []
      m2_scale_pspec = []
      if qdtype != jnp.bfloat16:
        m1_scale_pspec = _remove_leading_sharding_annotation(m1_pspec)
        m2_scale_pspec = _remove_leading_sharding_annotation(m2_pspec)

      local_stats_flat.append(
          LocalShardedParameterStats(
              QuantizedValue(param_pspec, [], [], jnp.bfloat16, False,
                             list(param.shape)),
              QuantizedValue(m1_pspec, [], m1_scale_pspec, qdtype, False,
                             list(param.shape)),
              QuantizedValue(m2_pspec, [], m2_scale_pspec, qdtype, False,
                             list(param.shape)),
              init_training_metrics_pspec(generate_training_metrics),
              index_start, sizes))

    local_stats = jax.tree_unflatten(treedef, local_stats_flat)
    global_stats = GlobalShardedParameterStats(partition_spec_for_statistics,
                                               partition_spec_for_statistics,
                                               pjit.PartitionSpec())
    count_pspec = pjit.PartitionSpec()
    return ShampooState(
        count=count_pspec, stats=ShardedShampooStats(global_stats, local_stats))

  def sharded_init_shape_and_dtype_fn(params):
    """Returns a parallel state tree with shape, dtype associated with state.
    Args:
      params: A pytree with params.
    """
    # Parallel lists of spec, and params.
    params_flat, treedef = jax.tree_flatten(params)
    assert params_flat
    # Step is replicated across cores.
    # None means cores.
    local_stats_flat = []
    num_statistics = 0
    for param in params_flat:
      param_clone = jnp.zeros(param.shape, dtype=param.dtype)
      preconditioner = preconditioner_from_params(param_clone)
      shapes = preconditioner.shapes_for_preconditioners()
      sizes = []

      index_start = num_statistics
      if not _skip_preconditioning(param):
        sizes = [s[0] for s in shapes]
        shapes = preconditioner.shapes_for_preconditioners()
        num_statistics += len(shapes)

      qdtype = quantized_dtype_for_momentum_buffers(param)
      m1_shape_and_dtype = [list(param.shape), param.dtype]
      m2_shape_and_dtype = [list(param.shape), param.dtype]
      m1_scale_shape_and_dtype = []
      m2_scale_shape_and_dtype = []
      if qdtype != jnp.bfloat16:
        m1_scale_shape_and_dtype = [list(param.shape)[1:], qdtype]
        m2_scale_shape_and_dtype = [list(param.shape)[1:], qdtype]

      diagonal_statistics_shape_and_dtype = [list(param.shape), param.dtype]
      local_stats_flat.append(
          LocalShardedParameterStats(
              QuantizedValue(diagonal_statistics_shape_and_dtype, [], [],
                             jnp.bfloat16, False, list(param.shape)),
              QuantizedValue(m1_shape_and_dtype, [], m1_scale_shape_and_dtype,
                             qdtype, False, list(param.shape)),
              QuantizedValue(m2_shape_and_dtype, [], m2_scale_shape_and_dtype,
                             qdtype, False, list(param.shape)),
              init_training_metrics_shapes(
                  len(sizes), generate_training_metrics),
              index_start,
              sizes,
          ))

    local_stats = jax.tree_unflatten(treedef, local_stats_flat)
    max_statistics_size = _max_statistics_size_from_params(params_flat)
    to_pad = -num_statistics % num_devices_for_pjit
    num_statistics += to_pad
    if num_statistics == 0:
      num_statistics = num_devices_for_pjit
      max_statistics_size = block_size
    statistics_shape = [
        num_statistics, max_statistics_size, max_statistics_size
    ]
    global_stats = GlobalShardedParameterStats([statistics_shape, jnp.bfloat16],
                                               [statistics_shape, jnp.bfloat16],
                                               [[num_statistics], jnp.int32])
    return ShampooState(
        count=[[], jnp.bfloat16],
        stats=ShardedShampooStats(global_stats, local_stats))


  def sharded_update_fn(grads, state, params):
    """Transform the input gradient and update all statistics in sharded mode.
    Args:
      grads: the gradient tensors for the parameters.
      state: a named tuple containing the state of the optimizer
      params: the parameters that should be updated.
    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    params_flat, treedef = jax.tree_flatten(params)
    grads_flat = treedef.flatten_up_to(grads)

    global_stats = state.stats.global_stats
    local_stats_flat = treedef.flatten_up_to(state.stats.local_stats)
    stats_flat = [
        _convert_to_parameter_stats(global_stats, local_stat)
        for local_stat in local_stats_flat
    ]
    new_stats_flat = jax.tree_map(
        lambda g, s, p: _compute_stats(g, s, p, state.count), grads_flat,
        stats_flat, params_flat)

    outputs = jax.tree_map(
        lambda g, s, p: _transform_grad(g, s, p, state.count), grads_flat,
        new_stats_flat, params_flat)
    updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

    updates = jax.tree_unflatten(treedef, updates_flat)
    # Create new local_stats
    new_local_stats_flat = [
        _convert_from_parameter_stats(new_stat, local_stat)
        for new_stat, local_stat in zip(new_stats_flat, local_stats_flat)
    ]

    max_size = global_stats.statistics.shape[1]
    new_padded_statistics = []
    padding_starts = []
    for stat in new_stats_flat:
      new_padded_statistics.extend(
          [pad_square_matrix(stat, max_size) for stat in stat.statistics])
      padding_starts.extend(
          [len(stat) for stat in stat.statistics])

    # Create global stats
    # TODO(rohananil): Preconditioner is not updated every step, so cost of
    # stack/pad can be obviated away.
    # Pad the statistics and preconditioner matrices to be a multiple of
    # num devices.
    # TODO(rohananil): Relax to only the size of the mesh axis where the dim
    # is split on.
    to_pad = -len(new_padded_statistics) % num_devices_for_pjit
    if not new_padded_statistics:
      to_pad = num_devices_for_pjit
      stat_dtype = jnp.bfloat16
    else:
      stat_dtype = new_padded_statistics[0].dtype

    new_padded_statistics.extend([
        jnp.eye(max_size, dtype=stat_dtype)
        for _ in range(to_pad)
    ])
    padding_starts += [0] * to_pad
    new_stacked_padded_statistics = jnp.stack(new_padded_statistics)
    new_stacked_padded_statistics = pjit.with_sharding_constraint(
        new_stacked_padded_statistics, statistics_partition_spec)
    stacked_padding_starts = jnp.array(padding_starts, jnp.int32)

    def _internal_inverse_pth_root_all():
      preconditioners, metrics = _matrix_inverse_pth_root_pjit(
          new_stacked_padded_statistics, global_stats.exponents,
          stacked_padding_starts, statistics_partition_spec)
      return preconditioners, metrics

    perform_step = state.count % preconditioning_compute_steps == 0

    if preconditioning_compute_steps == 1:
      new_preconditioners, metrics = _internal_inverse_pth_root_all()
    else:
      # Passing statistics instead of preconditioners as they are similarly
      # shaped tensors. Note statistics will be ignored as we are passing in
      # a large error value.
      preconditioners_init = new_stacked_padded_statistics
      n = new_stacked_padded_statistics.shape[0]
      metrics_init = cast(
          TrainingMetrics,
          init_training_metrics(n, generate_training_metrics=True))
      new_errors = jnp.ones_like(metrics_init.inverse_pth_root_errors) * (
          inverse_failure_threshold)
      metrics_init = metrics_init.replace(inverse_pth_root_errors=new_errors)
      init_state = [preconditioners_init, metrics_init]
      new_preconditioners, metrics = efficient_cond(
          perform_step, _internal_inverse_pth_root_all, init_state)

    if generate_training_metrics:
      new_local_stats_flat = _add_metrics_into_local_stats(
          new_local_stats_flat, metrics, ~perform_step)
    new_local_stats = jax.tree_unflatten(treedef, new_local_stats_flat)
    errors = metrics.inverse_pth_root_errors
    errors = errors.reshape((-1, 1, 1))
    predicate = jnp.logical_or(
        jnp.isnan(errors),
        errors >= inverse_failure_threshold).astype(new_preconditioners.dtype)
    # TODO(rohananil): Check for numerical instabilities.
    new_conditional_preconditioners = (
        predicate * global_stats.preconditioners +
        (1.0 - predicate) * new_preconditioners)
    new_global_stats = GlobalShardedParameterStats(
        new_stacked_padded_statistics, new_conditional_preconditioners,
        global_stats.exponents)
    new_shampoo_state = ShampooState(
        count=state.count + 1,
        stats=ShardedShampooStats(new_global_stats, new_local_stats))
    return updates, new_shampoo_state

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      preconditioner = preconditioner_from_params(param)
      statistics = []
      preconditioners = []
      if not _skip_preconditioning(param):
        shapes = preconditioner.shapes_for_preconditioners()
        statistics = [
            matrix_epsilon * jnp.eye(s[0], dtype=jnp.bfloat16) for s in shapes
        ]
        preconditioners = [jnp.eye(s[0], dtype=jnp.bfloat16) for s in shapes]

      diagonal_statistics = []
      if _graft_type_has_diagonal_statistics():
        diagonal_statistics = jnp.zeros_like(param)

      diagonal_momentum = _quantize_momentum(jnp.zeros_like(param))
      momentum = _quantize_momentum(jnp.zeros_like(param))

      return ParameterStats(
          _quantize_diagonal_statistics(diagonal_statistics),
          _maybe_quantize_statistics(statistics),
          _maybe_quantize_preconditioners(preconditioners), diagonal_momentum,
          momentum,
          init_training_metrics(len(statistics), generate_training_metrics))

    return ShampooState(
        count=jnp.zeros([], jnp.int32), stats=jax.tree_map(_init, params))

  def _skip_preconditioning(param):
    return len(param.shape) < skip_preconditioning_rank_lt or any(
        [s > skip_preconditioning_dim_size_gt for s in param.shape])


  def _compute_stats(grad, state, param, step):
    """Compute per-parameter statistics."""
    preconditioner = preconditioner_from_params(param)
    new_statistics = [[]] * len(state.statistics)
    w1 = beta2
    w2 = jnp.where(beta2 == 1.0, beta2, 1.0 - beta2)
    if not _skip_preconditioning(param):

      def compute_updated_statistics():
        return preconditioner.updated_statistics_from_grad(
            state.statistics,
            grad,
            w1=w1,
            w2=w2,
            to_float=_to_float,
            from_float=lambda x: _maybe_quantize_statistics([x])[0],
            precision=tensordot_precision,
        )

      if statistics_compute_steps > 1:
        perform_step = step % statistics_compute_steps == 0
        init_state = state.statistics
        new_statistics = list(
            efficient_cond(perform_step, compute_updated_statistics,
                           init_state))
      else:
        new_statistics = compute_updated_statistics()
    return ParameterStats(state.diagonal_statistics, new_statistics,
                          state.preconditioners, state.diagonal_momentum,
                          state.momentum, state.training_metrics)

  mi_pth_root = functools.partial(
      matrix_inverse_pth_root,
      ridge_epsilon=matrix_epsilon,
      precision=precision,
      relative_matrix_epsilon=relative_matrix_epsilon,
      lobpcg_topk_precondition=lobpcg_topk_precondition,
      lobpcg_max_iter=lobpcg_max_iter)


  def _matrix_inverse_pth_root_vmap(xs, ps, padding_starts):
    return jax.vmap(mi_pth_root)(xs, ps, padding_start=padding_starts)

  def _quantized_matrix_inverse_pth_root_vmap(qxs, qds, qbs, ps,
                                              padding_starts):

    def _quantized_to_float(qx, qd, qb):
      qv = QuantizedValue(qx, qd, qb, qx.dtype, True, list(qx.shape))
      return qv.to_float()

    def matrix_inverse_pth_root_wrapper(qx, qd, qb, p, padding_start):
      v = _quantized_to_float(qx, qd, qb)
      preconditioner, metrics = mi_pth_root(v, p, padding_start=padding_start)
      qp = QuantizedValue.from_float_value(preconditioner, qx.dtype, True)
      return qp.quantized, qp.diagonal, qp.bucket_size, metrics

    return jax.vmap(matrix_inverse_pth_root_wrapper)(qxs, qds, qbs, ps,
                                                     padding_starts)

  def _matrix_inverse_pth_root_pjit(xs,
                                    ps,
                                    padding_starts,
                                    statistics_partition_spec=None):
    # Partition the concatenated statistics matrix across all cores.
    pspec_for_partition = preconditioner_partition_spec
    partitioned_xs = pjit.with_sharding_constraint(xs, pspec_for_partition)
    if preconditioner_partition_spec:
      partitioned_ps_spec = pjit.PartitionSpec(preconditioner_partition_spec[0])
    else:
      partitioned_ps_spec = None
    partitioned_ps = pjit.with_sharding_constraint(
        ps, partitioned_ps_spec)
    partitioned_padding_starts = pjit.with_sharding_constraint(
        padding_starts, partitioned_ps_spec)  # paddings are scalars like ps.
    # Run matrix inverse pth root on each shard.
    partitioned_preconditioners, partitioned_metrics = (
        _matrix_inverse_pth_root_vmap(partitioned_xs, partitioned_ps,
                                      partitioned_padding_starts))
    # Reshard output to have the same PSpec as input. This is required to avoid
    # vmap seeing the full set of statistics.
    partitioned_preconditioners = pjit.with_sharding_constraint(
        partitioned_preconditioners, pspec_for_partition)
    # Recombine the outputs at each core.
    preconditioners = pjit.with_sharding_constraint(partitioned_preconditioners,
                                                    statistics_partition_spec)
    metrics = pjit.with_sharding_constraint(partitioned_metrics,
                                            pjit.PartitionSpec())
    return preconditioners, metrics

  def _pmap_compute_preconditioners(states, step, statistics,
                                    num_statistics_per_state, original_shapes,
                                    exponents, max_size, prev_preconditioners):
    """Computes preconditioners for given statistics in states in PMAP mode.
    Args:
      states: A list of optimizer states.
      step: Current step number
      statistics: A list of statistics for all variables (for every dim)
      num_statistics_per_state: Number of statistis per state to reconstruct
        output states.
      original_shapes: A list of shapes of the statistics.
      exponents: Exponent power to use for inverse-pth roots.
      max_size: Maximum dim of the statistics to pad.
      prev_preconditioners: Previously available preconditioner.
    Returns:
      New optimizer states after computing the preconditioner.
    """
    if batch_axis_name:
      num_devices = lax.psum(1, batch_axis_name)
    else:
      num_devices = 1
    num_statistics = len(statistics)
    # Pad statistics and exponents to next multiple of num_devices.
    packed_statistics = [
        pad_square_matrix(stat, max_size) for stat in statistics
    ]
    to_pad = -num_statistics % num_devices
    packed_statistics.extend([
        jnp.eye(max_size, dtype=packed_statistics[0].dtype)
        for _ in range(to_pad)
    ])
    exponents.extend([1 for _ in range(to_pad)])
    paddings = [len(stat) for stat in statistics] + [0] * to_pad

    if not packed_statistics:
      return states

    all_statistics = batch(packed_statistics, num_devices)
    all_exponents = batch(exponents, num_devices)
    all_paddings = batch(paddings, num_devices)

    def _internal_inverse_pth_root_all():
      if batch_axis_name:
        current_replica = lax.axis_index(batch_axis_name)
        preconditioners, metrics = _matrix_inverse_pth_root_vmap(
            all_statistics[current_replica], all_exponents[current_replica],
            all_paddings[current_replica])
        preconditioners = jax.lax.all_gather(preconditioners, batch_axis_name)
        metrics = jax.lax.all_gather(metrics, batch_axis_name)
        preconditioners_flat = unbatch(preconditioners)
        metrics_flat = jax.tree_map(unbatch, metrics)
      else:
        preconditioners, metrics = _matrix_inverse_pth_root_vmap(
            all_statistics[0], all_exponents[0], all_paddings[0])
        preconditioners_flat = unbatch(jnp.stack([preconditioners]))
        metrics = jax.tree_map(
            functools.partial(jnp.expand_dims, axis=0), metrics)
        metrics_flat = jax.tree_map(unbatch, metrics)

      return preconditioners_flat, metrics_flat

    perform_step = step % preconditioning_compute_steps == 0
    if preconditioning_compute_steps == 1:
      preconditioners_flat, metrics_flat = _internal_inverse_pth_root_all()
    else:
      # Passing statistics instead of preconditioners as they are similarly
      # shaped tensors. Note statistics will be ignored as we are passing in
      # a large error value.
      preconditioners_init = packed_statistics
      n = len(packed_statistics)
      metrics_init = jax.tree_map(
          lambda x: [x] * n,
          TrainingMetrics(inverse_pth_root_errors=inverse_failure_threshold))
      init_state = [preconditioners_init, metrics_init]
      preconditioners_flat, metrics_flat = efficient_cond(
          perform_step, _internal_inverse_pth_root_all, init_state)

    def _skip(error):
      condition = jnp.logical_or(
          jnp.isnan(error), error >= inverse_failure_threshold)
      return condition
      # return condition.astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
      return lax.cond(
          _skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_preconditioners_flat = []
    new_errors_flat = metrics_flat.inverse_pth_root_errors
    for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes,
                                       prev_preconditioners, new_errors_flat):
      new_preconditioners_flat.append(
          _select_preconditioner(error, p[:shape[0], :shape[1]], prev_p))

    assert len(states) == len(num_statistics_per_state)
    assert len(new_preconditioners_flat) == num_statistics
    assert len(new_errors_flat) == len(packed_statistics), (
        len(new_errors_flat), len(packed_statistics))
    assert len(new_errors_flat) == num_statistics + to_pad, (
        len(new_errors_flat), num_statistics, to_pad)

    # Add back empty preconditioners so we that we can set the optimizer state.
    preconditioners_for_states = []
    idx = 0
    metrics_for_states = []
    for num_statistics, state in zip(num_statistics_per_state, states):
      if num_statistics == 0:
        preconditioners_for_states.append([])
        metrics_for_states.append(
            init_training_metrics(0, generate_training_metrics))
      else:
        preconditioners_for_state = new_preconditioners_flat[idx:idx +
                                                             num_statistics]
        assert len(state.statistics) == len(preconditioners_for_state)
        preconditioners_for_states.append(preconditioners_for_state)

        if generate_training_metrics:
          # pylint:disable=cell-var-from-loop Used immediately.
          metrics_for_state = jax.tree_map(
              lambda x: jnp.stack(x[idx:idx + num_statistics]),
              metrics_flat,
              is_leaf=lambda x: isinstance(x, list))
          assert jax.tree_util.tree_all(
              jax.tree_map(lambda x: len(state.statistics) == len(x),
                           metrics_for_state))
          # If we skipped preconditioner computation, record old metrics.
          metrics_for_state = efficient_cond(perform_step,
                                             lambda: [metrics_for_state],
                                             [state.training_metrics])[0]
          # pylint:enable=cell-var-from-loop
        else:
          metrics_for_state = optax.MaskedNode()
        metrics_for_states.append(metrics_for_state)

        idx += num_statistics
    new_states = []
    for state, new_preconditioners, new_metrics in zip(
        states, preconditioners_for_states, metrics_for_states):
      # Note the preconditioner may have been skipped, but we still update the
      # metrics with the new error values; whether the preconditioner that's
      # actively being used is stale can be derived from the new_metrics
      # being greater than the failure threshold.
      new_states.append(
          ParameterStats(state.diagonal_statistics, state.statistics,
                         new_preconditioners, state.diagonal_momentum,
                         state.momentum, new_metrics))

    return new_states

  def _pmap_quantized_compute_preconditioners(states, step, statistics,
                                              num_statistics_per_state,
                                              original_shapes, exponents,
                                              max_size, prev_preconditioners):
    """Computes preconditioners for given statistics in states in PMAP mode.
    For quantization, each statistic is represented by three values:
      quantized matrix, diagonal, and bucket sizes, we run inverse pth-roots
      without ever recreating the original matrix in f32.
    Args:
      states: A list of optimizer states.
      step: Current step number
      statistics: A list of statistics for all variables (for every dim)
      num_statistics_per_state: Number of statistis per state to reconstruct
        output states.
      original_shapes: A list of shapes of the statistics.
      exponents: Exponent power to use for inverse-pth roots.
      max_size: Maximum dim of the statistics to pad.
      prev_preconditioners: Previously available preconditioner.
    Returns:
      New optimizer states after computing the preconditioner.
    """
    num_devices = lax.psum(1, batch_axis_name)
    num_statistics = len(statistics)
    quantized_dtype = quantized_dtype_for_second_moment_statistics_buffers()
    # Complexity here is around: shapes needing be statically shaped,
    # our custom quantization type requires a different type of packing.

    # Parallel tensors:
    # quantized [dxd]
    # diagonals [d] f32
    # bucket_sizes [d] f32
    packed_quantized_statistics = [
        pad_square_matrix(stat.quantized, max_size) for stat in statistics
    ]
    packed_quantized_diagonals = [
        pad_vector(stat.diagonal, max_size) for stat in statistics
    ]
    packed_quantized_bucket_sizes = [
        pad_vector(stat.bucket_size, max_size) for stat in statistics
    ]

    to_pad = -num_statistics % num_devices
    padded_eye = jnp.eye(max_size, dtype=jnp.bfloat16)
    quantized_eye = QuantizedValue.from_float_value(padded_eye, quantized_dtype,
                                                    True)
    packed_quantized_statistics.extend(
        [quantized_eye.quantized for _ in range(to_pad)])
    packed_quantized_diagonals.extend(
        [quantized_eye.diagonal for _ in range(to_pad)])
    packed_quantized_bucket_sizes.extend(
        [quantized_eye.bucket_size for _ in range(to_pad)])
    exponents.extend([1 for _ in range(to_pad)])
    paddings = [len(stat.quantized) for stat in statistics] + [0] * to_pad

    if not packed_quantized_statistics:
      return states

    all_quantized_statistics = batch(packed_quantized_statistics, num_devices)
    all_quantized_diagonals = batch(packed_quantized_diagonals, num_devices)
    all_quantized_bucket_sizes = batch(packed_quantized_bucket_sizes,
                                       num_devices)
    all_exponents = batch(exponents, num_devices)
    all_paddings = batch(paddings, num_devices)

    def _internal_inverse_pth_root_all():
      current_replica = lax.axis_index(batch_axis_name)
      (quantized_preconditioners, quantized_diagonals, quantized_bucket_sizes,
       metrics) = (
           _quantized_matrix_inverse_pth_root_vmap(
               all_quantized_statistics[current_replica],
               all_quantized_diagonals[current_replica],
               all_quantized_bucket_sizes[current_replica],
               all_exponents[current_replica], all_paddings[current_replica]))
      quantized_preconditioners = jax.lax.all_gather(quantized_preconditioners,
                                                     batch_axis_name)
      quantized_diagonals = jax.lax.all_gather(quantized_diagonals,
                                               batch_axis_name)
      quantized_bucket_sizes = jax.lax.all_gather(quantized_bucket_sizes,
                                                  batch_axis_name)
      metrics = jax.lax.all_gather(metrics, batch_axis_name)
      quantized_preconditioners_flat = unbatch(quantized_preconditioners)
      quantized_diagonals_flat = unbatch(quantized_diagonals)
      quantized_bucket_sizes_flat = unbatch(quantized_bucket_sizes)
      metrics_flat = jax.tree_map(unbatch, metrics)
      return (quantized_preconditioners_flat, quantized_diagonals_flat,
              quantized_bucket_sizes_flat, metrics_flat)

    perform_step = step % preconditioning_compute_steps == 0
    if preconditioning_compute_steps == 1:
      (quantized_preconditioners_flat, quantized_diagonals_flat,
       quantized_bucket_sizes_flat, metrics_flat) = (
           _internal_inverse_pth_root_all())
    else:
      # Passing statistics instead of preconditioners as they are similarly
      # shaped tensors. Note statistics will be ignored as we are passing in
      # a large error value.
      quantized_preconditioners_init = packed_quantized_statistics
      quantized_diagonals_init = packed_quantized_diagonals
      quantized_bucket_sizes_init = packed_quantized_bucket_sizes
      n = len(quantized_preconditioners_init)
      metrics_init = jax.tree_map(
          lambda x: [x] * n,
          TrainingMetrics(inverse_pth_root_errors=inverse_failure_threshold))
      init_state = [
          quantized_preconditioners_init, quantized_diagonals_init,
          quantized_bucket_sizes_init, metrics_init
      ]
      (quantized_preconditioners_flat, quantized_diagonals_flat,
       quantized_bucket_sizes_flat, metrics_flat) = (
           efficient_cond(perform_step, _internal_inverse_pth_root_all,
                          init_state))

    def _skip(error):
      condition = jnp.logical_or(
          jnp.isnan(error), error >= inverse_failure_threshold)
      return condition.astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
      return lax.cond(
          _skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_quantized_preconditioners_flat = []
    new_quantized_diagonals_flat = []
    new_quantized_bucket_sizes_flat = []
    new_errors_flat = metrics_flat.inverse_pth_root_errors
    for p, d, b, shape, prev_p, error in zip(quantized_preconditioners_flat,
                                             quantized_diagonals_flat,
                                             quantized_bucket_sizes_flat,
                                             original_shapes,
                                             prev_preconditioners,
                                             new_errors_flat):
      new_quantized_preconditioners_flat.append(
          _select_preconditioner(error, p[:shape[0], :shape[1]],
                                 prev_p.quantized))
      new_quantized_diagonals_flat.append(
          _select_preconditioner(error, d[:shape[0]], prev_p.diagonal))
      new_quantized_bucket_sizes_flat.append(
          _select_preconditioner(error, b[:shape[0]], prev_p.bucket_size))

    assert len(states) == len(num_statistics_per_state)
    assert len(new_quantized_preconditioners_flat) == num_statistics
    assert len(new_quantized_diagonals_flat) == num_statistics
    assert len(new_quantized_bucket_sizes_flat) == num_statistics

    # Add back empty preconditioners so we that we can set the optimizer state.
    preconditioners_for_states = []
    metrics_for_states = []
    idx = 0
    for num_statistics, state in zip(num_statistics_per_state, states):
      if num_statistics == 0:
        preconditioners_for_states.append([])
        metrics_for_states.append(
            init_training_metrics(0, generate_training_metrics))
      else:
        quantized_preconditioners_for_state = new_quantized_preconditioners_flat[
            idx:idx + num_statistics]
        quantized_diagonals_for_state = new_quantized_diagonals_flat[
            idx:idx + num_statistics]
        quantized_bucket_sizes_for_state = new_quantized_bucket_sizes_flat[
            idx:idx + num_statistics]

        if generate_training_metrics:
          # pylint:disable=cell-var-from-loop Used immediately.
          metrics_for_state = jax.tree_map(
              lambda x: jnp.stack(x[idx:idx + num_statistics]),
              metrics_flat,
              is_leaf=lambda x: isinstance(x, list))

          assert len(
              state.statistics) == len(quantized_preconditioners_for_state)
          assert len(state.statistics) == len(quantized_diagonals_for_state)
          assert len(state.statistics) == len(quantized_bucket_sizes_for_state)
          assert jax.tree_util.tree_all(
              jax.tree_map(lambda x: len(state.statistics) == len(x),
                           metrics_for_state))

          # If we skipped preconditioner computation, record old metrics.
          metrics_for_state = efficient_cond(perform_step,
                                             lambda: [metrics_for_state],
                                             [state.training_metrics])[0]
          # pylint:enable=cell-var-from-loop
        else:
          metrics_for_state = optax.MaskedNode()

        quantized_preconditioners = []
        for qv, qd, qb in zip(quantized_preconditioners_for_state,
                              quantized_diagonals_for_state,
                              quantized_bucket_sizes_for_state):
          quantized_preconditioners.append(
              QuantizedValue(qv, qd, qb, qv.dtype, True, list(qv.shape)))
        preconditioners_for_states.append(quantized_preconditioners)
        metrics_for_states.append(metrics_for_state)
        idx += num_statistics
    new_states = []
    for state, new_preconditioners, new_metrics in zip(
        states, preconditioners_for_states, metrics_for_states):
      # Note the preconditioner may have been skipped, but we still update the
      # metrics with the new error values; whether the preconditioner that's
      # actively being used is stale can be derived from the new_metrics
      # being greater than the failure threshold.
      new_states.append(
          ParameterStats(state.diagonal_statistics, state.statistics,
                         new_preconditioners, state.diagonal_momentum,
                         state.momentum, new_metrics))

    return new_states

  def _pjit_compute_preconditioners(states, step, statistics,
                                    num_statistics_per_state, original_shapes,
                                    exponents, max_size, prev_preconditioners):
    """Computes preconditioners for given statistics in states in PJIT mode.
    Args:
      states: A list of optimizer states.
      step: Current step number
      statistics: A list of statistics for all variables (for every dim)
      num_statistics_per_state: Number of statistis per state to reconstruct
        output states.
      original_shapes: A list of shapes of the statistics.
      exponents: Exponent power to use for inverse-pth roots.
      max_size: Maximum dim of the statistics to pad.
      prev_preconditioners: Previously available preconditioner.
    Returns:
      New optimizer states after computing the preconditioner.
    """
    num_statistics = len(statistics)
    to_pad = -num_statistics % num_devices_for_pjit
    padded_statistics = [
        pad_square_matrix(stat, max_size) for stat in statistics
    ]
    padded_statistics.extend([
        jnp.eye(max_size, dtype=padded_statistics[0].dtype)
        for _ in range(to_pad)
    ])
    exponents.extend([1 for _ in range(to_pad)])
    paddings = [len(stat) for stat in statistics] + [0] * to_pad
    all_statistics = jnp.stack(padded_statistics)
    all_exponents = jnp.stack(exponents)
    all_paddings = jnp.stack(paddings)

    def _internal_inverse_pth_root_all():
      preconditioners, metrics = _matrix_inverse_pth_root_pjit(
          all_statistics, all_exponents, all_paddings)
      b1 = preconditioners.shape[0]

      def split(batched_values):
        return [
            jnp.squeeze(v)
            for v in jnp.split(batched_values, indices_or_sections=b1, axis=0)
        ]

      return split(preconditioners), jax.tree_map(split, metrics)

    if preconditioning_compute_steps == 1:
      preconditioners_flat, metrics_flat = _internal_inverse_pth_root_all()
    else:
      # Passing statistics instead of preconditioners as they are similarly
      # shaped tensors. Note statistics will be ignored as we are passing in
      # a large init value for error.
      preconditioners_init = padded_statistics
      n = len(padded_statistics)
      metrics_init = jax.tree_map(
          lambda x: [x] * n,
          TrainingMetrics(inverse_pth_root_errors=inverse_failure_threshold))
      init_state = [preconditioners_init, metrics_init]
      perform_step = step % preconditioning_compute_steps == 0
      preconditioners_flat, metrics_flat = efficient_cond(
          perform_step, _internal_inverse_pth_root_all, init_state)

    def _skip(error):
      condition = jnp.logical_or(
          jnp.isnan(error), error >= inverse_failure_threshold)
      return condition.astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
      return lax.cond(
          _skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_preconditioners_flat = []
    new_errors_flat = metrics_flat.inverse_pth_root_errors
    for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes,
                                       prev_preconditioners, new_errors_flat):
      new_preconditioners_flat.append(
          _select_preconditioner(error.inverse_pth_root_errors,
                                 p[:shape[0], :shape[1]], prev_p))

    assert len(states) == len(num_statistics_per_state)
    assert len(new_preconditioners_flat) == num_statistics

    # Add back empty preconditioners so we that we can set the optimizer state.
    preconditioners_for_states = []
    metrics_for_states = []
    idx = 0
    for num_statistics, state in zip(num_statistics_per_state, states):
      if num_statistics == 0:
        preconditioners_for_states.append([])
        metrics_for_states.append(
            init_training_metrics(0, generate_training_metrics))
      else:
        preconditioners_for_state = new_preconditioners_flat[idx:idx +
                                                             num_statistics]
        assert len(state.statistics) == len(preconditioners_for_state)
        preconditioners_for_states.append(preconditioners_for_state)

        if generate_training_metrics:
          # pylint:disable=cell-var-from-loop Used immediately.
          metrics_for_state = jax.tree_map(
              lambda x: jnp.stack(x[idx:idx + num_statistics]),
              metrics_flat,
              is_leaf=functools.partial(isinstance, list))
          assert jax.tree_util.tree_all(
              jax.tree_map(lambda x: len(state.statistics) == len(x),
                           metrics_for_state))
          # pylint:enable=cell-var-from-loop
        else:
          metrics_for_state = optax.MaskedNode()
        metrics_for_states.append(metrics_for_state)
        idx += num_statistics

    new_states = []
    for state, new_preconditioners, new_metrics in zip(
        states, preconditioners_for_states, metrics_for_states):
      new_states.append(
          ParameterStats(state.diagonal_statistics, state.statistics,
                         new_preconditioners, state.diagonal_momentum,
                         state.momentum, new_metrics))

    return new_states

  def _compute_preconditioners(states, params, step):
    """Computes preconditioners for given statistics in states.
    Args:
      states: A list of optimizer states.
      params: A list of params.
      step: Current step number
    Returns:
      New optimizer states after computing the preconditioner.
    """
    statistics = []
    num_statistics_per_state = []
    original_shapes = []
    exponents = []
    max_size = 0
    prev_preconditioners = []

    for state, param in zip(states, params):
      num_statistics = len(state.statistics)
      num_statistics_per_state.append(num_statistics)
      original_shapes_for_state = []
      if num_statistics > 0:
        preconditioner = preconditioner_from_params(param)
        for statistic in state.statistics:
          exponents.append(preconditioner.exponent_for_preconditioner(
          ) if exponent_override == 0 else exponent_override)
          original_shapes_for_state.append(statistic.shape)
          max_size = max(max_size, statistic.shape[0])

        statistics.extend(state.statistics)
        prev_preconditioners.extend(state.preconditioners)
        original_shapes.extend(original_shapes_for_state)

    if not shard_optimizer_states:
      # Quantization is only enabled if batch_axis_name is not set.
      quantized_dtype = quantized_dtype_for_second_moment_statistics_buffers()

      if quantized_dtype == jnp.bfloat16:
        return _pmap_compute_preconditioners(states, step, statistics,
                                             num_statistics_per_state,
                                             original_shapes, exponents,
                                             max_size, prev_preconditioners)
      else:
        return _pmap_quantized_compute_preconditioners(
            states, step, statistics, num_statistics_per_state, original_shapes,
            exponents, max_size, prev_preconditioners)

    else:
      return _pjit_compute_preconditioners(states, step, statistics,
                                           num_statistics_per_state,
                                           original_shapes, exponents, max_size,
                                           prev_preconditioners)

  def _transform_grad(grad, state, param, step):
    """Transform per-parameter gradients."""
    preconditioner = preconditioner_from_params(param)
    sgd_update = grad
    new_diagonal_statistics = state.diagonal_statistics.to_float()

    if (graft_type == GraftingType.ADAGRAD or
        graft_type == GraftingType.ADAGRAD_NORMALIZED):

      scaled_grad = grad
      if graft_type == GraftingType.ADAGRAD_NORMALIZED:
        scaled_grad = grad / (jnp.linalg.norm(grad) + 1e-16)

      new_diagonal_statistics = (
          state.diagonal_statistics.to_float() + jnp.square(scaled_grad))
      adagrad_update = scaled_grad / (
          jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)
      grafting_update = adagrad_update
    elif (graft_type == GraftingType.RMSPROP or
          graft_type == GraftingType.RMSPROP_NORMALIZED):

      scaled_grad = grad
      if graft_type == GraftingType.RMSPROP_NORMALIZED:
        scaled_grad = grad / (jnp.linalg.norm(grad) + 1e-16)

      w1 = beta2
      w2 = jnp.where(beta2 == 1.0, beta2, 1.0 - beta2)

      new_diagonal_statistics = (
          w1 * state.diagonal_statistics.to_float() +
          w2 * jnp.square(scaled_grad))
      rmsprop_update = scaled_grad / (
          jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)

      if clip_by_scaled_gradient_norm:
        scaled_grad_norm = jnp.linalg.norm(rmsprop_update) / (
            jnp.sqrt(float(rmsprop_update.size)))
        clipping_denom = jnp.maximum(
            1., scaled_grad_norm / clip_by_scaled_gradient_norm)
        rmsprop_update /= clipping_denom

      grafting_update = rmsprop_update
    elif graft_type == GraftingType.SGD:
      grafting_update = sgd_update
    else:
      grafting_update = jnp.ones_like(sgd_update) * jnp.sign(sgd_update)

    lr = learning_rate
    if callable(learning_rate):
      lr = learning_rate(step)

    preconditioner_multiplier = lr if not decoupled_learning_rate else 1.0
    grafting_update = grafting_update * preconditioner_multiplier

    precond_grad = grad
    if not _skip_preconditioning(param):
      precond_grad = preconditioner.preconditioned_grad(
          precond_grad,
          _maybe_dequantize_preconditioners(state.preconditioners))
    else:
      precond_grad = grafting_update

    grafting_update_norm = jnp.linalg.norm(grafting_update)
    precond_grad_norm = jnp.linalg.norm(precond_grad)

    multiplier = (grafting_update_norm / (precond_grad_norm + 1e-16))
    shampoo_update = precond_grad * multiplier

    shampoo_update_with_wd = shampoo_update
    grafting_update_with_wd = grafting_update

    if weight_decay != 0 and not decoupled_weight_decay:
      shampoo_update_with_wd = shampoo_update + weight_decay * param
      grafting_update_with_wd = grafting_update + weight_decay * param

    w = (1.0 - beta1) if moving_average_for_momentum else 1.0

    shampoo_update_with_wd_momentum = (
        state.momentum.to_float() * beta1 + w * shampoo_update_with_wd)

    grafting_update_with_wd_momentum = (
        state.diagonal_momentum.to_float() * beta1 +
        w * grafting_update_with_wd)

    run_shampoo = (step >= start_preconditioning_step).astype(
        grafting_update_with_wd_momentum.dtype)

    momentum_update = (
        run_shampoo * shampoo_update_with_wd_momentum +
        (1.0 - run_shampoo) * grafting_update_with_wd_momentum)

    wd_update = (
        run_shampoo * shampoo_update_with_wd +
        (1.0 - run_shampoo) * grafting_update_with_wd)

    nesterov_momentum_update = momentum_update

    if nesterov:
      nesterov_momentum_update = w * wd_update + beta1 * momentum_update

    if weight_decay != 0 and decoupled_weight_decay:
      nesterov_momentum_update = (
          nesterov_momentum_update + lr * weight_decay * param)

    momentum_multiplier = lr if decoupled_learning_rate else 1.0
    transformed_update = -1.0 * momentum_multiplier * nesterov_momentum_update

    new_diagonal_momentum = grafting_update_with_wd_momentum
    new_momentum = shampoo_update_with_wd_momentum

    param_stats = ParameterStats(
        _quantize_diagonal_statistics(new_diagonal_statistics),
        state.statistics, state.preconditioners,
        _quantize_momentum(new_diagonal_momentum),
        _quantize_momentum(new_momentum), state.training_metrics)

    return transformed_update, param_stats

  def update_fn(grads, state, params):
    """Transform the input gradient and update all statistics.
    Args:
      grads: the gradient tensors for the parameters
        and any custom gradients for preconditioners.
      state: a named tuple containing the state of the optimizer
      params: the parameters that should be updated.
    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    params_flat, treedef = jax.tree_flatten(params)
    stats_flat = treedef.flatten_up_to(state.stats)
    grads_flat = treedef.flatten_up_to(grads)
    stats_grads = grads_flat


    new_stats_flat = jax.tree_map(
        lambda g, s, p: _compute_stats(g, s, p, state.count), stats_grads,
        stats_flat, params_flat)

    new_stats_flat = _compute_preconditioners(new_stats_flat, params_flat,
                                              state.count)
    outputs = jax.tree_map(
        lambda g, s, p: _transform_grad(g, s, p, state.count), grads_flat,
        new_stats_flat, params_flat)
    updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

    updates = jax.tree_unflatten(treedef, updates_flat)
    new_stats = jax.tree_unflatten(treedef, new_stats_flat)

    new_state = ShampooState(count=state.count + 1, stats=new_stats)
    return updates, new_state


  if shard_optimizer_states:
    # Hijacks the init_fn signature so we can return an OptState with
    # appropriate init_fns.
    opt_init_fn = sharded_init_fn
    def _init_fns(unused_params):
      return InitFnState(
          init_fn=opt_init_fn,
          pspec_fn=sharded_init_partition_spec_fn,
          shape_and_dtype_fn=sharded_init_shape_and_dtype_fn)

    opt_update_fn = sharded_update_fn
    return optax.GradientTransformation(_init_fns, opt_update_fn)
  else:
    return optax.GradientTransformation(init_fn, update_fn)




# flags.DEFINE_float('beta1', 0.9, help='Beta1')
# flags.DEFINE_float('beta2', 0.999, help='Beta2')
# flags.DEFINE_float('lr', 0.0001, help='Learning rate')
# flags.DEFINE_float('eps', 1e-8, help='eps')
# flags.DEFINE_integer('batch_size',
#                      1000, help='Batch size.')
# flags.DEFINE_integer('model_size_multiplier',
#                      1, help='Multiply model size by a constant')
# flags.DEFINE_integer('model_depth_multiplier',
#                      1, help='Multiply model depth by a constant')
# flags.DEFINE_integer('warmup_epochs', 5, help='Warmup epochs')
# flags.DEFINE_integer('epochs', 100, help='#Epochs')
# flags.DEFINE_integer('t', 20, help='preconditioner computation frequency')
# flags.DEFINE_enum('dtype', 'float32', ['float32', 'bfloat16'], help='dtype')
# flags.DEFINE_enum('optimizer', 'shampoo', ['sgd', 'momentum', 'nesterov', 'adagrad',
#   'rmsprop', 'tds', 'shampoo', 'diag_sonew'], help='optimizer')
# FLAGS = flags.FLAGS


#51.24
parser = argparse.ArgumentParser()
parser.add_argument('--beta1', type=float, default=8.37282e-1, help='beta1')
parser.add_argument('--beta2', type=float, default=8.30400e-1, help='beta2')
parser.add_argument('--lr', type=float, default=5.03785e-3, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=1000, help='batch_size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--eps', type=float, default=3.10165e-4, help='eps')
parser.add_argument('--t', type=int, default=150, help="preconditioner update frequency")
parser.add_argument('--warmup_epochs', type=int, default=5, help="warmup epochs")
parser.add_argument('--model_depth_multiplier', type=int, default=1, help="model_depth_multiplier")
parser.add_argument('--model_size_multiplier', type=int, default=1, help="model_size_multiplier")
parser.add_argument('--optimizer', type=str, default="shampoo", help="optimizer")
parser.add_argument('--dtype', type=str, default="bfloat16", help="float32 or bfloat16")

FLAGS = parser.parse_args(args=[])

mode = 1
if mode == 1:
  FLAGS.beta1 = 8.1261e-1
  FLAGS.beta2 = 7.53646e-1
  FLAGS.lr = 4.6012e-3
  FLAGS.eps = 2.4290e-4
  FLAGS.t = 80
if mode == 2:
  FLAGS.beta1 = 9.01865e-1
  FLAGS.beta2 = 9.19932e-1
  FLAGS.lr = 5.1034e-3
  FLAGS.eps = 2.5968e-4
  FLAGS.t = 20
if mode == 3:
  FLAGS.beta1 = 0.9
  FLAGS.beta2 = 9.17082e-1
  FLAGS.lr = 2.29684e-3
  FLAGS.eps = 9.87363e-4
  FLAGS.t = 1




class Autoencoder(nn.Module):
  enc_hidden_states: List[int]
  dec_hidden_states: List[int]
  dtype: Any
  param_dtype: Any

  @nn.compact
  def __call__(self, x):
    for i in range(len(self.enc_hidden_states)):
      x = nn.Dense(features = self.enc_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      if i<len(self.enc_hidden_states)-1:
        x = nn.tanh(x)
    for i in range(len(self.dec_hidden_states)):
      x = nn.Dense(features = self.dec_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      x = nn.tanh(x)
    x = nn.Dense(features = 784,
                 kernel_init=jnn.initializers.glorot_uniform(),
                 dtype=self.dtype, param_dtype=self.param_dtype)(x)
    return x

  def __hash__(self):
    return id(self)


def get_optimizer(opt, learning_rate):
  print("using optimizer:", opt)
  if opt=="shampoo":
    return distributed_shampoo(
      learning_rate=learning_rate,
      block_size=2048,
      beta1=FLAGS.beta1,
      beta2=FLAGS.beta2,
      diagonal_epsilon=1e-12,
      matrix_epsilon=FLAGS.eps,
      weight_decay=0.0,
      start_preconditioning_step=25,
      preconditioning_compute_steps=FLAGS.t,
      statistics_compute_steps=1,
      best_effort_shape_interpretation=True,
      graft_type=4,
      nesterov=False,
      best_effort_memory_usage_reduction=True,
      inverse_failure_threshold=0.1,
      moving_average_for_momentum=True,
      skip_preconditioning_dim_size_gt=4096,
      clip_by_scaled_gradient_norm=None,
      batch_axis_name=None)
  else:
      raise NotImplementedError

def create_train_state(params, model, opt, learning_rate):
  """Creates initial `TrainState`."""
  tx = get_optimizer(opt, learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

# Training epoch
@partial(jax.jit, static_argnums=0)
def train_step(model, state, x):
  def loss_fn(params):
    logits = model.apply(params, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, x).mean(0).sum()
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

@partial(jax.jit, static_argnums=0)
def eval_step(model, state, x):
  logits = model.apply(state.params, x)
  loss = optax.sigmoid_binary_cross_entropy(logits, x)
  return loss.astype(jnp.float32).mean(0).sum()

def train_epoch(state, model, train_ds, batch_size, epoch, rng, lrVec):
  start_epoch = time.time()
  train_ds_size = len(train_ds)
  steps_per_epoch = train_ds_size // batch_size
  print("epoch:", epoch,"and lr going to be used:", lrVec[epoch])

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    train_x = train_ds[perm]
    state, loss = train_step(model, state, train_x)
    print("loss:", loss, loss.dtype)
    batch_metrics.append(loss.item())

  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = np.mean(batch_metrics_np)
  print("this epoch took time:", time.time()-start_epoch)

  # print('train epoch: %d, loss: %.4f' % (epoch, epoch_metrics_np))

  return state


def main(argv):
  start = time.time()
  #Get random keys
  rng = jax.random.PRNGKey(0)
  rng, key1 = jax.random.split(rng)
  rng, key2 = jax.random.split(rng)
  rng, key3 = jax.random.split(rng)

  #Get dtype
  if FLAGS.dtype=="float32":
    dtype = jnp.float32
  elif FLAGS.dtype=="bfloat16":
    dtype = jnp.bfloat16
  else:
      raise NotImplementedError

  #Generate data
  (train_inputs, _), (test_inputs, test_labels) = mnist.load_data()
  train_inputs = jnp.array(train_inputs).astype(jnp.float32)
  test_inputs = jnp.array(test_inputs).astype(jnp.float32)

  # Rescale input images to [0, 1]
  train_inputs = jnp.reshape(train_inputs, [-1, 784]) / 255.0
  test_inputs = jnp.reshape(test_inputs, [-1, 784]) / 255.0

  train_inputs = train_inputs.astype(dtype)
  test_inputs = test_inputs.astype(dtype)

  num_train_examples = train_inputs.shape[0]
  num_test_examples = test_inputs.shape[0]
  print('MNIST dataset:')
  print('Num train examples: ' + str(num_train_examples))
  print('Num test examples: ' + str(num_test_examples))

  batch_size = FLAGS.batch_size

  encoder_sizes = [1000] +  [500] * FLAGS.model_depth_multiplier + [250, 30]
  decoder_sizes = [250] +  [500] * FLAGS.model_depth_multiplier + [1000]

  encoder_sizes = [FLAGS.model_size_multiplier * e for e in encoder_sizes]
  decoder_sizes = [FLAGS.model_size_multiplier * e for e in decoder_sizes]
  encoder_decoder_sizes = encoder_sizes, decoder_sizes

  input_image_batch = np.random.normal(size=(batch_size,784))
  input_image_batch = jnp.array(input_image_batch).astype(dtype)

  #Set learning rate schedule array
  num_epochs = FLAGS.epochs
  warmup_epochs = FLAGS.warmup_epochs
  lr = FLAGS.lr
  lrVec = np.concatenate([np.linspace(0,lr,warmup_epochs),
                          np.linspace(lr,0,num_epochs-warmup_epochs+2)[1:-1]],
                         axis=0)
  lrVec = jnp.array(lrVec).astype(dtype)
  def autoencoder_shedule(lrVec):
    def schedule(count):
      bucket = count//60
      return lrVec[bucket]
    return schedule

  train_loss_val_=[]
  model = Autoencoder(encoder_sizes, decoder_sizes, dtype=dtype, param_dtype=dtype)
  params = model.init(key3, input_image_batch)
  state = create_train_state(params, model, FLAGS.optimizer, autoencoder_shedule(lrVec))
  print("Initialized model and optimizer!")

  for i in range(num_epochs):
    rng, key = jax.random.split(rng)
    state = train_epoch(state, model, train_inputs, FLAGS.batch_size, i, key, lrVec)
    train_loss_val = eval_step(model, state, train_inputs)
    train_loss_val_.append(train_loss_val)
    print("epoch: " + str(i) +", train_loss_val: " + str(train_loss_val))
    print("")
  print("training total time is:", time.time()-start)


if __name__ == '__main__':
  print("FLAGS:", FLAGS)
  app.run(main)
  # main()