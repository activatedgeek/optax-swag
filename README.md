# SWAG in Optax

This package implements [SWAG](https://arxiv.org/abs/1902.02476) 
as an [Optax](https://optax.readthedocs.io/) transform to allow
usage with [JAX](https://jax.readthedocs.io/).

## Installation

For now, the only available mode of installation is directly from source as
```
pip install git+https://github.com/activatedgeek/optax-swag.git
```

**NOTE**: A PyPI package will be available soon.

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

# License

Apache 2.0