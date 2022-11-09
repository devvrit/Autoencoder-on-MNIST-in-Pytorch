import jax
import jax.numpy as jnp                # JAX NumPy
from jax import nn as jnn              # JAX nn
import argparse
import time
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

from absl import app
from functools import partial
from absl import flags
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
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
import kfac_jax
from jax import lax
from jax.experimental import pjit
from jax.experimental.sparse import linalg
import jax.numpy as jnp
import numpy as np
import optax

import os
os.environ['TF_CUDNN_DETERMINISTIC']='1'


flags.DEFINE_float('beta1', 0.9, help='Beta1 for Adam')
flags.DEFINE_float('beta2', 9.4925e-1, help='Beta2 for Adam')
flags.DEFINE_float('lr', 3.5338e-3, help='Learning rate')
flags.DEFINE_float('eps', 1e-8, help='eps')
flags.DEFINE_integer('batch_size',
                     1000, help='Batch size.')
flags.DEFINE_integer('model_size_multiplier',
                     1, help='Multiply model size by a constant')
flags.DEFINE_integer('model_depth_multiplier',
                     1, help='Multiply model depth by a constant')
flags.DEFINE_integer('warmup_epochs', 5, help='Warmup epochs')
flags.DEFINE_integer('t', 80, help='preconditioner computation frequency')
flags.DEFINE_integer('epochs', 100, help='#Epochs')
flags.DEFINE_enum('optimizer', 'distributed_shampoo',[
    'sgd', 'momentum', 'nesterov', 'adam', 'rmsprop', 'adagrad', 'diag_quic',
    'distributed_shampoo', 'tds'], 'Which optimizer to use')
FLAGS = flags.FLAGS


class Autoencoder(nn.Module):
  enc_hidden_states: List[int]
  dec_hidden_states: List[int]
  dtype: Any
  param_dtype: Any

  @nn.compact
  def __call__(self, x):
    for i in range(len(self.enc_hidden_states)):
      x = nn.Dense(features=self.enc_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      if i < len(self.enc_hidden_states)-1:
        x = nn.tanh(x)
    for i in range(len(self.dec_hidden_states)):
      x = nn.Dense(features=self.dec_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      x = nn.tanh(x)
    x = nn.Dense(features=784,
                 kernel_init=jnn.initializers.glorot_uniform(),
                 dtype=self.dtype, param_dtype=self.param_dtype)(x)
    return x

  def __hash__(self):
    return id(self)


def train_epoch(params, opt_state, model, train_ds, batch_size, epoch, key,
                rng, lrvec, optimizer):
  train_ds_size = len(train_ds)
  steps_per_epoch = train_ds_size // batch_size
  print('epoch:', epoch, ' and lr going to be used:', lrvec[epoch])

  perms = jax.random.permutation(key, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  for i, perm in enumerate(perms):
    train_x = train_ds[perm]
    rng, key = jax.random.split(rng)
    params, opt_state, stats = optimizer.step(
        params, opt_state, key, batch=train_x, global_step_int=i)
    print(i, stats)

  return params, opt_state, stats, rng

def main(argv):
  start = time.time()
  rng = jax.random.PRNGKey(0)
  rng, key1 = jax.random.split(rng)
  rng, key2 = jax.random.split(rng)
  rng, key3 = jax.random.split(rng)
  del key2

  dtype = jnp.float32

  # Generate data
  (train_inputs, _), (test_inputs, _) = mnist.load_data()
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
  # encoder_decoder_sizes = encoder_sizes, decoder_sizes

  input_image_batch = np.random.normal(size=(batch_size, 784))
  input_image_batch = jnp.array(input_image_batch).astype(dtype)

  # Set learning rate schedule array
  num_epochs = FLAGS.epochs
  warmup_epochs = FLAGS.warmup_epochs
  lr = FLAGS.lr
  lrvec = np.concatenate([np.linspace(0, lr, warmup_epochs),
                          np.linspace(lr, 0, num_epochs-warmup_epochs+2)[1:-1]],
                         axis=0)
  lrvec = jnp.array(lrvec).astype(dtype)
  def autoencoder_shedule(lrvec):
    def schedule(count):
      bucket = count//60
      return lrvec[bucket]
    return schedule

  # train_loss_val_=[]
  model = Autoencoder(encoder_sizes, decoder_sizes, dtype=dtype,
                      param_dtype=dtype)
  params = model.init(key3, input_image_batch)
  def loss_fn(params, x):
    logits = model.apply(params, x)
    kfac_jax.register_sigmoid_cross_entropy_loss(logits, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, x).mean(0).sum()
    return loss

  optimizer = kfac_jax.Optimizer(value_and_grad_func=
                                 jax.value_and_grad(loss_fn),
                                 l2_reg=0,
                                 learning_rate_schedule=
                                 autoencoder_shedule(lrvec),
                                 momentum_schedule=lambda x: FLAGS.beta1,
                                 use_adaptive_damping=True,
                                 initial_damping=1.0,
                                 curvature_ema=FLAGS.beta2,
                                 multi_device=False)
  opt_state = optimizer.init(params, key1, input_image_batch)
  print('Initialized model and optimizer!')

  for i in range(num_epochs):
    start_epoch = time.time()
    rng, key = jax.random.split(rng)
    params, opt_state, stats, rng = train_epoch(params, opt_state, model,
                                                train_inputs, FLAGS.batch_size,
                                                i, key, rng, lrvec, optimizer)
    print('this epoch took time:', time.time() - start_epoch)
    print('')
  print('total training time:', time.time() - start)

if __name__ == '__main__':
  app.run(main)