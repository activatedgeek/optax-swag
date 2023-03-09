import jax
import jax.numpy as jnp


@jax.jit
def tree_split(key, ref_tree):
    treedef = jax.tree_util.tree_structure(ref_tree)
    key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)
    return key, jax.tree_util.tree_unflatten(treedef, key_list)


@jax.jit
def sample_tree_diag_gaussian(key, mean_tree, var_tree):
    _, key_tree = tree_split(key, mean_tree)
    def _sample_param(key, mu, var):
        return mu + jnp.sqrt(var) * jax.random.normal(key, mu.shape, mu.dtype)
    return jax.tree_util.tree_map(_sample_param, key_tree, mean_tree, var_tree)
