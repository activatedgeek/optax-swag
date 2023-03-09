# SWAG in Optax

This package implements [SWAG](https://arxiv.org/abs/1902.02476) 
as an [Optax](https://optax.readthedocs.io/) transform to allow
usage with [JAX](https://jax.readthedocs.io/).

## Installation

For now, the only available mode of installation is directly from source as
```
pip install git+https://github.com/activatedgeek/optax-swag.git
```

**TODO**: A PyPI package will be available soon.

## Usage

To start updating the iterate statistics, use [chaining](https://optax.readthedocs.io/en/latest/api.html#chain) as

```python
import optax
from optax_swag import swag_diag

optimizer = optax.chain(
    ...  ## Other optimizer and transform config.
    swag_diag(freq)  ## Always add as the last transform.
)
```

The [SWAGState](./optax_swag/transform.py#L8) object can be accessed from
the optimizer state list for downstream usage.

**NOTE**: If using with non-parameter variables like BatchNorm statistics, make sure to update such
values with `SWAGState.params` (the mean parameter from SWA).

### Sampling

A reference code to generate samples from the collected statistics
in `SWAGState` object is provided below.

```python
import jax
import jax.numpy as jnp

from optax_swag import sample_tree_diag_gaussian

opt_state = # Reference to a SWAGState object.

rng = jax.random.PRNGKey(42)
swa_samples = 10

swa_mean = opt_state.mean
swa_params2 = opt_state.params2
swa_var = jax.tree_util.tree_map(lambda mu, p2: jnp.square(mu) - p2,
                                    swa_mean, swa_params2)

rng, *samples_rng = jax.random.split(rng, 1 + swa_samples)
swag_sample_params = jax.vmap(sample_tree_diag_gaussian, in_axes=(0, None, None))(
    jnp.array(samples_rng), swa_mean, swa_var)
```

`swag_sample_params` are PyTree objects that can be used for downstream evaluation.

# License

Apache 2.0