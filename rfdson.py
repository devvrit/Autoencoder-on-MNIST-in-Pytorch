from typing import Any, List, NamedTuple, Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax
from functools import partial
from optax._src import utils
from optax._src import combine
from optax._src import base
from optax._src import alias
ScalarOrSchedule = Union[float, base.Schedule]

def scale_by_learning_rate(
    learning_rate: ScalarOrSchedule,
    flip_sign: bool = True) -> optax.GradientTransformation:
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)

def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(lambda g, t: (1 - decay) * (g.reshape(-1)**order) + decay * t,
                      updates, moments)

# @partial(jax.jit, static_argnums=4)
def update_b(b, alpha, m):
  if b.shape[0]<=m:
    return b, alpha
  u, s, vh = jnp.linalg.svd(b, False)
  s = jnp.diag(s)
  b = jnp.matmul(jnp.sqrt(s[:m, :m]**2 - s[m, m]**2 * jnp.eye(m)), vh[:m])
  alpha = alpha + (s[m, m]**2)/2
  return b, alpha


class PreconditionRfdSONState(NamedTuple):
  """State for the rfdSON optimizer."""
  count: chex.Array # shape=(), dtype=jnp.int32
  mu: optax.Updates
  b: optax.Updates
  alpha: optax.Updates

#Pre conditioning by rfdson preconditioner
def precondition_by_rfdson(beta1: float = 0.9,
                           mu_t: float = 1/8,
                           compute: int = 1,
                           m: int = 2,
                           alpha_init: float = 0.1,
                           debias: bool = True) -> optax.GradientTransformation:
  """Preconditioning  by rfdson structure."""
  def init_fn(params):
    return PreconditionRfdSONState(
      count=jnp.zeros([], jnp.int32),
      mu=jax.tree_map(lambda p: jnp.zeros(len(p.reshape(-1))), params),
      b=jax.tree_map(lambda p: jnp.zeros([]), params),
      alpha=jax.tree_map(lambda p: jnp.array(alpha_init), params)
    )

  def update_fn(updates, state, params=None):
    count = state.count + 1
    alpha = state.alpha
    b = state.b

    mu = _update_moment(updates, state.mu, beta1, 1)
    nu_t = 1/(count)
    updates_hat = jax.tree_map(lambda g: jnp.sqrt(mu_t + nu_t)*g.reshape(-1), updates)
    if len(jax.tree_util.tree_leaves(b)[0].shape) == 0:
      b = jax.tree_map(lambda g: g.reshape((1, -1)), updates_hat)
    else:
      b = jax.tree_map(lambda b_, g: jnp.concatenate((b_, g.reshape((1,-1))), axis=0), b,
                       updates_hat)
    # if count%compute == 0:
    temp = jax.tree_map(lambda b_, a: update_b(b_, a, m), b, alpha)
    b = jax.tree_map(lambda g, t: t[0], mu, temp)
    alpha = jax.tree_map(lambda g, t: t[1], mu, temp)
    update = jax.tree_map(lambda g, a, b_: g/a - (
      jnp.matmul(b_.T, jnp.matmul(jnp.linalg.inv(
      jnp.eye(b_.shape[0]) + jnp.matmul(b_, b_.T)/a), jnp.matmul(b_, g) )))/(a**2),
      mu, alpha, b
    )
    updates = jax.tree_map(lambda u, g: u.reshape(g.shape), update, updates)
    return updates, PreconditionRfdSONState(count, mu, b, alpha)

  return optax.GradientTransformation(init_fn, update_fn)


def rfdson(learning_rate: ScalarOrSchedule,
           beta1: float = 0.9,
           mu_t: float = 1/8,
           compute: int = 1,
           m: int = 2,
           alpha_init: float = 0.1,
           weight_decay: float = 0.0) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_rfdson(beta1=beta1, mu_t=mu_t,
                             compute=compute, m=m,
                             alpha_init=alpha_init),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )
