"""Sparse preconditioners."""
from typing import NamedTuple, Union

import chex
import jax
import jax.numpy as jnp
import optax


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(lambda g, t: (1 - decay) * (g**order) + decay * t,
                      updates, moments)


def _update_moment_vmap(updates, moments, decay, order):
  return (1-decay) * (updates**order) + decay * moments


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


class PreconditionDiagonalSONewState(NamedTuple):
  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu: optax.Updates


def precondition_by_diag_sonew(
    beta1: float = 0.999,
    beta2: float = 0.9,
    eps: float = 1e-8,
    adam_grafting: bool = False,
    debias: bool = True) -> optax.GradientTransformation:
  """Stochastic Online Newton preconditioner."""

  def init_fn(params):
    return PreconditionDiagonalSONewState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, beta1, 1)
    nu = _update_moment(updates, state.nu, beta2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, beta1, count)
    nu_hat = nu if not debias else _bias_correction(nu, beta2, count)
    updates = jax.tree_map(lambda m, v: m / (v + eps), mu_hat, nu_hat)
    if adam_grafting:
      adam_updates = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat,
                                  nu_hat)
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au)/(jnp.linalg.norm(u)+1e-12)) * u,
          updates, adam_updates)
    return updates, PreconditionDiagonalSONewState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(
    learning_rate: ScalarOrSchedule,
    flip_sign: bool = True) -> optax.GradientTransformation:
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def diag_sonew(learning_rate: ScalarOrSchedule,
               beta1: float = 0.9,
               beta2: float = 0.999,
               eps: float = 1e-8,
               weight_decay: float = 0.0,
               adam_grafting: bool = False) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_diag_sonew(beta1=beta1, beta2=beta2, eps=eps,
                                 adam_grafting=adam_grafting),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )


def _update_nu(updates, nu_e, nu_d, beta2):
  """Compute the exponential moving average of the tridiagonal structure of the moment."""
  updates = jax.tree_map(lambda g: g/(jnp.sqrt(jnp.linalg.norm(g))+1e-16),
                         updates)
  nu_d = jax.tree_map(lambda g, t: (1 - beta2) * (g**2) + beta2 * t, updates,
                      nu_d)
  nu_e = jax.tree_map(lambda g, t: (1 - beta2) * (g[:-1] * g[1:]) + beta2 * t,
                      updates, nu_e)
  return nu_e, nu_d


def _update_nu_vmap(update, nu_e, nu_d, beta2):
  """Compute the exponential moving average of the tridiagonal structure of the moment."""
  update = update/(jnp.sqrt(jnp.linalg.norm(update))+1e-16)
  nu_d = (1-beta2) * (update**2) + beta2 * nu_d
  nu_e = (1-beta2) * (update[:-1] * update[1:]) + beta2 * nu_e
  return nu_e, nu_d


class PreconditionTriDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  diag: optax.Updates


def precondition_by_tds(beta1: float = 0.9,
                        beta2: float = 0.999,
                        eps: float = 1e-8,
                        graft_type: int = 0,
                        graft_eps: float = 1e-8,
                        transpose: bool = True,
                        debias: bool = True) -> optax.GradientTransformation:
  """Preconditioning  by tri-diagonal structure."""
  def diag_init_fn_vmap(param):
    return jnp.zeros(len(param.reshape(-1)))

  def subdiag_init_fn_vmap(param):
    return jnp.zeros(len(param.reshape(-1))-1)

  def init_fn(params):
    diag = None
    params = jax.tree_map(lambda g: g.reshape(-1, 1) if len(g.shape) == 1  # pylint: disable=g-long-lambda
                          else g, params)
    if graft_type == 4 or graft_type == 1:  # Adam/Normalized rmsprop grafting
      diag = jax.tree_map(lambda g: jax.vmap(diag_init_fn_vmap(g.T)) if  # pylint: disable=g-long-lambda
                          transpose else jax.vmap(diag_init_fn_vmap(g)), params)
    return PreconditionTriDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_map(lambda g: jax.vmap(subdiag_init_fn_vmap(g.T)) if  # pylint: disable=g-long-lambda
                          transpose else jax.vmap(subdiag_init_fn_vmap(g)),
                          params),
        nu_d=jax.tree_map(lambda g: jax.vmap(diag_init_fn_vmap(g.T)) if  # pylint: disable=g-long-lambda
                          transpose else jax.vmap(diag_init_fn_vmap(g)),
                          params),
        diag=diag)

  def precondition_grad(d, e, g):
    updates = d*g
    updates = updates.at[1:].set(updates[1:] + g[:-1]*e)
    updates = updates.at[:-1].set(updates[:-1] + g[1:]*e)
    return updates

  def graft_adam(update, adam_update):
    update = jnp.linalg.norm(adam_update)/(jnp.sqrt(
        jnp.linalg.norm(update))+1e-16) * update

  def update_fn(updates, state, params=None):
    diag = state.diag
    updates = jax.tree_map(lambda g: g.reshape(-1, 1) if len(g.shape) == 1  # pylint: disable=g-long-lambda
                           else g, updates)
    # updates = jax.tree_map(lambda g: g/(jnp.linalg.norm(g)+1e-16), updates)
    updates_hat = jax.tree_map(lambda g: jax.vmap(diag_init_fn_vmap(g.T))  # pylint: disable=g-long-lambda
                               if transpose else jax.vmap(diag_init_fn_vmap(g)),
                               updates)

    mu = _update_moment(updates, state.mu, beta1, 1)
    nu_e, nu_d = jax.tree_map(lambda g, u, d: jax.vmap(_update_nu_vmap(g, u, d,  # pylint: disable=g-long-lambda
                                                                       beta2)),
                              updates_hat, state.nu_e, state.nu_d, beta2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _bias_correction(mu, beta1, count) if debias else mu
    nu_hat_e = _bias_correction(nu_e, beta2, count) if debias else nu_e
    nu_hat_d = _bias_correction(nu_d, beta2, count) if debias else nu_d

    temp = jax.tree_map(lambda d, e:  # pylint: disable=g-long-lambda
                        jax.vmap(tridiagonal.tridiag_kfac(d, e, eps)),
                        nu_hat_d, nu_hat_e)
    pre_d = jax.tree_map(lambda h, g: g[0], nu_hat_d, temp)
    pre_e = jax.tree_map(lambda h, g: g[1], nu_hat_e, temp)

    mu_hat_flat = jax.tree_map(lambda g: jax.vmap(diag_init_fn_vmap(g.T))  # pylint: disable=g-long-lambda
                               if transpose else jax.vmap(diag_init_fn_vmap(g)),
                               mu_hat)

    updates = jax.tree_map(lambda d, e, g: jax.vmap(precondition_grad(d, e, g)),
                           pre_d, pre_e, mu_hat_flat)

    if graft_type == 1:
      diag = jax.tree_map(lambda g, d:  # pylint: disable=g-long-lambda
                          jax.vmap(_update_moment_vmap(g, d, beta2, 2)),
                          updates_hat, diag)
      adam_updates = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) +  graft_eps),
                                  mu_hat_flat, diag)
      updates = jax.tree_map(lambda u, au: jax.vmap(graft_adam(u, au)),
                             updates, adam_updates)

    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T  # pylint: disable=g-long-lambda
        if transpose else mf.reshape(m.shape), updates, params)
    # adam_updates = jax.tree_map(
    #     lambda mf, m: mf.reshape(m.T.shape).T  # pylint: disable=g-long-lambda
    #     if transpose else mf.reshape(m.shape), adam_updates, params)
    # updates = jax.tree_map(lambda u, au: au if len(u.shape) <= 1 else u,
    #                        updates, adam_updates)
    return updates, PreconditionTriDiagonalState(
        count=count, mu=mu, nu_e=nu_e, nu_d=nu_d, diag=diag)

  return optax.GradientTransformation(init_fn, update_fn)


def tds(learning_rate: ScalarOrSchedule,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        graft_type: int = 0,
        graft_eps: float = 1e-8,
        weight_decay: float = 0.0,
        transpose: bool = True) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_tds(beta1=beta1, beta2=beta2, eps=eps,
                          graft_type=graft_type,
                          graft_eps=graft_eps,
                          transpose=transpose),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )


def _update_nu_banded(updates, nu_e, nu_d, beta2, lams):
  nu_d = jax.tree_map(lambda g, t, lam: (1-beta2) * (g**2)/lam + beta2 * t,
                      updates, nu_d, lams)
  def update_band(g, band, b, lam):
    for i in range(b):
      band = band.at[:-(i+1), i].set((1-beta2)*(g[:-(i+1)]*g[i+1:])/lam +
                                     beta2*band[:-(i+1), i])
    return band
  nu_e = jax.tree_map(lambda g, t, lam: update_band(g, t, t.shape[-1], lam),
                      updates, nu_e, lams)
  return nu_e, nu_d


class PreconditionBandedDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  diag: optax.Updates


def precondition_by_bds(beta1: float = 0.9,
                        beta2: float = 0.999,
                        eps: float = 1e-8,
                        graft_eps: float = 1e-8,
                        graft_type: int = 0,
                        transpose: bool = True,
                        b: int = 3,
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
        diag=diag)

  def update_fn(updates, state, params):
    del params
    diag = state.diag
    updates_hat = jax.tree_map(
        lambda g: g.T.reshape(-1) if transpose else g.reshape(-1), updates)
    mu = _update_moment(updates, state.mu, beta1, 1)

    count = state.count + jnp.array(1, dtype=jnp.int32)

    # lams = jax.tree_map(lambda d, g: jnp.sqrt(jnp.sum(d))+1e-16, state.nu_d,
    #                     updates_hat)
    # lams = jax.tree_map(lambda d, g: jnp.linalg.norm(g)+1e-16, state.nu_d,
    #                     updates_hat)
    lams = jax.tree_map(lambda d, g: 1, state.nu_d, updates_hat)

    nu_e, nu_d = _update_nu_banded(updates_hat, state.nu_e,
                                   state.nu_d, beta2, lams)
    mu_hat = mu if not debias else _bias_correction(mu, beta1, count)
    nu_hat_e = nu_e if not debias else _bias_correction(nu_e, beta2, count)
    nu_hat_d = nu_d if not debias else _bias_correction(nu_d, beta2, count)

    mu_hat_flat = jax.tree_map(lambda m: m.T.reshape(-1)
                               if transpose else m.reshape(-1), mu_hat)
    updates = jax.tree_map(lambda d, e, g:
                           tridiagonal.bandedUpdates(d, e, eps, g),
                           nu_hat_d, nu_hat_e, mu_hat_flat)
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
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u)+1e-16)) * u,
          updates, updates_hat)
    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T  
        if transpose else mf.reshape(m.shape), updates, mu_hat)
    return updates, PreconditionBandedDiagonalState(
        count=count, mu=mu, nu_e=nu_e, nu_d=nu_d, diag=diag)

  return optax.GradientTransformation(init_fn, update_fn)


def bds(learning_rate: ScalarOrSchedule,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        graft_eps: float = 1e-8,
        graft_type: int = 0,
        weight_decay: float = 0.0,
        b: int = 4,
        transpose: bool = True) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_bds(beta1=beta1, beta2=beta2, eps=eps,
                          graft_type=graft_type, graft_eps=graft_eps, b=b,
                          transpose=transpose),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )

