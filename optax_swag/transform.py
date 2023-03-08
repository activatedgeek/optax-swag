from typing import NamedTuple
import chex
import optax
import jax.numpy as jnp
import jax


class SWAGState(NamedTuple):
  """State for SWAG mean and non-centered variance."""
  k: chex.Array  # Step count.
  n: chex.Array  # Iterate count using for running stats.
  params: optax.Params  # Running mean of iterates.
  params_var: optax.Params  # Running non-centered variance of iterates.


def swag_diag(update_freq):
  def init_fn(params):
    return SWAGState(
        k=jnp.zeros([], jnp.int32),
        n=jnp.zeros([], jnp.int32),
        params=params,
        params_var=jax.tree_util.tree_map(lambda t: jnp.square(t), params))

  def update_fn(updates, state, params):
    k = (state.k + 1) % update_freq
    update_mask = k == 0

    n = state.n + 1 * update_mask

    next_params = jax.tree_util.tree_map(lambda u, p: u * update_mask + p, updates, params)
    next_mean = jax.tree_util.tree_map(lambda mu, np: (n * mu + np) / (n + 1), 
                                       state.params, next_params)
    next_var = jax.tree_util.tree_map(lambda v, np: (n * v + jnp.square(np)) / (n + 1),
                                      state.params_var, next_params)

    return updates, SWAGState(k=k, n=n, params=next_mean, params_var=next_var)

  return optax.GradientTransformation(init_fn, update_fn)