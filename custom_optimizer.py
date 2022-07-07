from typing import Any, List, NamedTuple, Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax

# from quic_numpy.tridiagFirstOrder import *
import tridiagonal_quic
import jax.numpy as jnp
from optax._src import utils
from optax._src import combine
from optax._src import base
from optax._src import alias
ScalarOrSchedule = Union[float, base.Schedule]

class PreconditionTriDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
    
def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_multimap(lambda g, t: (1 - decay) * (g**order) + decay * t,
                           updates, moments)

def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def _update_nu(updates, nu_e, nu_d, beta2):
  """Compute the exponential moving average of the tridiagonal structure of the moment."""
  nu_d = jax.tree_multimap(lambda g, t: (1-beta2) * (g**2) + beta2 * t,
                           updates, nu_d)
  nu_e = jax.tree_multimap(lambda g, t: (1-beta2) * (g[:-1]*g[1:]) + beta2 * t,
                           updates, nu_e)
  return nu_e, nu_d

# def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
#   m = -1 if flip_sign else 1
#   if callable(learning_rate):
#     return transform.scale_by_schedule(lambda count: m * learning_rate(count))
#   return transform.scale(m * learning_rate)

#Pre conditioning by tri diagonal structure
def precondition_by_tds(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    debias: bool = True) -> optax.GradientTransformation:

  def init_fn(params):
    return PreconditionTriDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu = jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_multimap(lambda g: jnp.zeros(len(g.reshape(-1))-1), params),
        nu_d=jax.tree_multimap(lambda g: jnp.zeros(len(g.reshape(-1))), params))
  
  def update_fn(updates, state, params):
    updates_hat = jax.tree_multimap(lambda g: g.reshape(-1), updates)
    mu = _update_moment(updates, state.mu, b1, 1)
    nu_e, nu_d = _update_nu(updates_hat, state.nu_e, state.nu_d, b2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat_e = nu_e if not debias else _bias_correction(nu_e, b2, count)
    nu_hat_d = nu_d if not debias else _bias_correction(nu_d, b2, count)

    temp = jax.tree_multimap(lambda d, e:
                             tridiagonal_quic.tridiagKFAC(d+eps,e),
                             nu_hat_d, nu_hat_e)
    pre_d = jax.tree_multimap(lambda h, g: g[0], nu_hat_d, temp)
    pre_e = jax.tree_multimap(lambda h, g: g[1], nu_hat_d, temp)
    mu_hat_flat = jax.tree_multimap(lambda m: jnp.append(jnp.append(0.0, m.reshape(-1)),0.0), mu_hat)
    pre_e = jax.tree_multimap(lambda g: jnp.append(jnp.append(0.0, g), 0.0), pre_e)
    updates = jax.tree_multimap(lambda mf, m, a, b:
                                (mf[:-2]*a[:-1] + mf[1:-1]*b + mf[2:]*a[1:]).reshape(m.shape),
                                mu_hat_flat, mu_hat, pre_e, pre_d)
    return updates, PreconditionTriDiagonalState(count=count, mu=mu, nu_e=nu_e,
                                                 nu_d=nu_d)

  return optax.GradientTransformation(init_fn, update_fn)

def tds(learning_rate: ScalarOrSchedule, b1=0.9, b2=0.99, eps=1e-8):
    return combine.chain(
      precondition_by_tds(
          b1=b1, b2=b2, eps=eps),
        alias._scale_by_learning_rate(learning_rate),
    )
