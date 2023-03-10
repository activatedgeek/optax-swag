# SWAG in Optax

[![PyPI version](https://badge.fury.io/py/optax-swag.svg)](https://pypi.org/project/optax-swag/)

This package implements [SWAG](https://arxiv.org/abs/1902.02476) 
as an [Optax](https://optax.readthedocs.io/) transform to allow
usage with [JAX](https://jax.readthedocs.io/).

## Installation

Install from `pip` as:
```shell
pip install optax-swag
```

To install the latest directly from source, run
```shell
pip install git+https://github.com/activatedgeek/optax-swag.git
```

## Usage

To start updating the iterate statistics, use [chaining](https://optax.readthedocs.io/en/latest/api.html#chain) as

```python
import optax
from optax_swag import swag

optimizer = optax.chain(
    ...  ## Other optimizer and transform config.
    swag(freq, rank)  ## Always add as the last transform.
)
```

The [SWAGState](./optax_swag/state.py#L22) object can be accessed from
the optimizer state list for downstream usage.

### Sampling

A reference code to generate samples from the collected statistics
is provided below.

```python
import jax
import jax.numpy as jnp

from optax_swag import sample_swag

swa_opt_state = # Reference to a SWAGState object from the optimizer.
n_samples = 10

rng = jax.random.PRNGKey(42)
rng, *samples_rng = jax.random.split(rng, 1 + n_samples)

swag_sample_params = jax.vmap(sample_swag, in_axes=(0, None))(
    jnp.array(samples_rng), swa_opt_state)
```

The resulting `swag_sample_params` can now be used for downstream evaluation.

**NOTE**: Make sure to update non-parameter variables (e.g. BatchNorm running statistics) for each generated sample.

# License

Apache 2.0