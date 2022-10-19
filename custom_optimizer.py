from typing import Any, List, NamedTuple, Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax

# from quic_numpy.tridiagFirstOrder import *
import tridiagonal_quic as tridiagonal
import jax.numpy as jnp
from optax._src import utils
from optax._src import combine
from optax._src import base
from optax._src import alias
ScalarOrSchedule = Union[float, base.Schedule]

    
def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(lambda g, t: (1 - decay) * (g**order) + decay * t,
                           updates, moments)

def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def _update_nu(updates, nu_e, nu_d, beta2):
  """Compute the exponential moving average of the tridiagonal structure of the moment."""
  nu_d = jax.tree_map(lambda g, t: (1-beta2) * (g**2) + beta2 * t,
                           updates, nu_d)
  nu_e = jax.tree_map(lambda g, t: (1-beta2) * (g[:-1]*g[1:]) + beta2 * t,
                           updates, nu_e)
  return nu_e, nu_d


def _update_nu_banded(updates, nu_e, nu_d, beta2):
  nu_d = jax.tree_map(lambda g, t: (1-beta2) * (g**2) + beta2 * t,
                      updates, nu_d)
  def update_band(g, band, b):
    for i in range(b):
      band = band.at[:-(i+1), i].set((1-beta2)*(g[:-(i+1)]*g[i+1:]) +
                                     beta2*band[:-(i+1), i])
    return band
  nu_e = jax.tree_map(lambda g, t: update_band(g, t, t.shape[-1]), updates,
                      nu_e)
  return nu_e, nu_d


def scale_by_learning_rate(
    learning_rate: ScalarOrSchedule,
    flip_sign: bool = True) -> optax.GradientTransformation:
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)



class PreconditionTriDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  diag: optax.Updates

#Pre conditioning by tri diagonal structure
def precondition_by_tds(
    b1: float = 0.999,
    b2: float = 0.9,
    eps: float = 1e-8,
    graft_eps: float = 1e-8,
    transpose: bool = True,
    graft_type: int = 0,
    debias: bool = True) -> optax.GradientTransformation:

  def init_fn(params):
    diag = None
    if graft_type == 2:
      diag = jax.tree_map(lambda g: jnp.zeros(len(g.reshape(-1)),
                                              dtype=g.dtype), params)
    return PreconditionTriDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_map(lambda g: jnp.zeros(len(g.reshape(-1))-1,
                                              dtype=g.dtype), params),
        nu_d=jax.tree_map(lambda g: jnp.zeros(len(g.reshape(-1)),
                                              dtype=g.dtype), params),
        diag=diag)

  def update_fn(updates, state, params):
    del params
    diag = state.diag
    updates_hat = jax.tree_map(lambda g: g.T.reshape(-1)
                               if transpose else g.reshape(-1), updates)
    mu = _update_moment(updates, state.mu, b1, 1)
    nu_e, nu_d = _update_nu(updates_hat, state.nu_e, state.nu_d, b2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat_e = nu_e if not debias else _bias_correction(nu_e, b2, count)
    nu_hat_d = nu_d if not debias else _bias_correction(nu_d, b2, count)

    temp = jax.tree_map(lambda d, e:
                        tridiagonal.tridiagKFAC(d, e, eps=eps),
                        nu_hat_d, nu_hat_e)
    pre_d = jax.tree_map(lambda h, g: g[0], nu_hat_d, temp)
    pre_e = jax.tree_map(lambda h, g: g[1], nu_hat_e, temp)

    mu_hat_flat = jax.tree_map(lambda m: m.T.reshape(-1)
                               if transpose else m.reshape(-1), mu_hat)
    # Multiply gradient with diagonal
    updates = jax.tree_map(lambda m, a: m*a, mu_hat_flat, pre_d)
    # updates[i] = updates[i] + gradient[i-1]*pre_e[i], for i>0
    updates = jax.tree_map(lambda u, m, a: u.at[1:].set(u[1:]+m[:-1]*a),
                           updates, mu_hat_flat, pre_e)
    # updates[i] = updates[i] + gradient[i+1]*pre_e[i], for i<n-1
    updates = jax.tree_map(lambda u, m, a: u.at[:-1].set(u[:-1]+m[1:]*a),
                           updates, mu_hat_flat, pre_e)
    if graft_type == 1:
      adam_updates = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + graft_eps),
                                  mu_hat_flat, nu_hat_d)
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u)+1e-12)) * u,
          updates, adam_updates)
    elif graft_type == 2:
      updates_hat = jax.tree_map(lambda g: g/(jnp.linalg.norm(g)+1e-16),
                                 updates_hat)
      diag = _update_moment(updates_hat, diag, b2, 2)
      updates_hat = jax.tree_map(lambda g, d: g/(jnp.sqrt(d)+graft_eps),
                                 updates_hat, diag)
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u)+1e-12)) * u,
          updates, updates_hat)
    # reshape them to the original param shapes
    updates = jax.tree_map(lambda mf, m: mf.reshape(m.T.shape).T
                           if transpose else mf.reshape(m.shape),
                           updates, mu_hat)

    # An alternaative way to multiply with tridiagonal matrix is below
    '''
    mu_hat_flat = jax.tree_map(lambda m: jnp.append(jnp.append(0.0, m.reshape(-1)),0.0), mu_hat)
    pre_e = jax.tree_map(lambda g: jnp.append(jnp.append(0.0, g), 0.0), pre_e)
    updates = jax.tree_map(lambda mf, m, a, b:
                                (mf[:-2]*a[:-1] + mf[1:-1]*b + mf[2:]*a[1:]).reshape(m.shape),
                                mu_hat_flat, mu_hat, pre_e, pre_d)
    # print(pre_e, pre_d)
    '''
    return updates, PreconditionTriDiagonalState(count=count, mu=mu, nu_e=nu_e,
                                                 nu_d=nu_d, diag=diag)

  return optax.GradientTransformation(init_fn, update_fn)


def tds(learning_rate: ScalarOrSchedule, b1=0.9, b2=0.99, eps=1e-8,
        weight_decay: float = 0.0, graft_type: int = 0, graft_eps: float = 1e-8,
        transpose=True):
  return combine.chain(
      precondition_by_tds(
          b1=b1, b2=b2, eps=eps, transpose=transpose, graft_eps=graft_eps,
          graft_type=graft_type),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )


class PreconditionBandedDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  ind: optax.Updates
  diag: optax.Updates

def precondition_by_bds(beta1: float = 0.9,
                        beta2: float = 0.999,
                        eps: float = 1e-8,
                        graft_eps: float = 1e-8,
                        graft_type: int = 0,
                        transpose: bool = True,
                        ridge_epsilon: float = 1e-12,
                        b: int = 3,
                        innerIters = 15,
                        debias: bool = True) -> optax.GradientTransformation:
  def init_fn(params):
    diag = None
    if graft_type == 2:
      diag = jax.tree_map(lambda g: jnp.zeros(len(g.reshape(-1)),
                                              dtype=g.dtype), params)
    return PreconditionBandedDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_map(lambda g: jnp.zeros((len(g.reshape(-1)), b),
                                              dtype=g.dtype), params),
        nu_d=jax.tree_map(lambda g: jnp.zeros(len(g.reshape(-1)),
                                              dtype=g.dtype), params),
        ind=jax.tree_map(lambda g:
                         tridiagonal.createInd(len(g.reshape(-1)), b), params),
        diag=diag)

  def update_fn(updates, state, params):
    del params
    diag = state.diag
    updates_hat = jax.tree_map(
        lambda g: g.T.reshape(-1) if transpose else g.reshape(-1), updates)
    mu = _update_moment(updates, state.mu, beta1, 1)
    nu_e, nu_d = _update_nu_banded(updates_hat, state.nu_e,
                                   state.nu_d, beta2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, beta1, count)
    nu_hat_e = nu_e if not debias else _bias_correction(nu_e, beta2, count)
    nu_hat_d = nu_d if not debias else _bias_correction(nu_d, beta2, count)

    mu_hat_flat = jax.tree_map(lambda m: m.T.reshape(-1)
                               if transpose else m.reshape(-1), mu_hat)
    updates = jax.tree_map(lambda d, e, g, ind:
                           tridiagonal.bandedUpdates(d, e,
                                                     ind, eps,
                                                     innerIters, g),
                           nu_hat_d, nu_hat_e, mu_hat_flat, state.ind)
    if graft_type == 1:
      adam_updates = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + graft_eps),
                                  mu_hat_flat, nu_hat_d)
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u)+1e-12)) * u,
          updates, adam_updates)
    elif graft_type == 2:
      updates_hat = jax.tree_map(lambda g: g/(jnp.linalg.norm(g)+1e-16),
                                 updates_hat)
      diag = _update_moment(updates_hat, diag, beta2, 2)
      updates_hat = jax.tree_map(lambda g, d: g/(jnp.sqrt(d)+graft_eps),
                                 updates_hat, diag)
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u)+1e-12)) * u,
          updates, updates_hat)
    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T  
        if transpose else mf.reshape(m.shape), updates, mu_hat)
    return updates, PreconditionBandedDiagonalState(
        count=count, mu=mu, nu_e=nu_e, nu_d=nu_d, ind=state.ind, diag=diag)

  return optax.GradientTransformation(init_fn, update_fn)


def bds(learning_rate: ScalarOrSchedule,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        graft_eps: float = 1e-8,
        graft_type: int = 0,
        weight_decay: float = 0.0,
        ridge_epsilon: float = 1e-12,
        b: int = 3,
        transpose: bool = True) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_bds(beta1=beta1, beta2=beta2, eps=eps,
                          graft_type=graft_type, innerIters=20,
                          graft_eps=graft_eps, ridge_epsilon=ridge_epsilon, b=b,
                          transpose=transpose),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )

