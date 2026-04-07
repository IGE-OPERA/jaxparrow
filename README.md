# jaxparrow

![Python](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/meom-group/jaxparrow/master/.github/workflows/python-package.yml&label=Python&query=$.jobs.build.strategy.matrix["python-version"])
[![PyPi](https://img.shields.io/badge/dynamic/xml?url=https://pypi.org/rss/project/jaxparrow/releases.xml&label=PyPi&query=/rss/channel/item[1]/title)](https://pypi.org/project/jaxparrow/)
![Tests](https://github.com/meom-group/jaxparrow/actions/workflows/python-package.yml/badge.svg)
[![Docs](https://app.readthedocs.org/projects/jaxparrow/badge/)](https://jaxparrow.readthedocs.io/)
[![DOI](https://zenodo.org/badge/702998298.svg)](https://zenodo.org/badge/latestdoi/702998298)

`jaxparrow` implements a novel approach based on a minimization-based formulation to compute the inversion of the cyclogeostrophic balance.

It leverages the power of [`JAX`](https://jax.readthedocs.io/en/latest/), to efficiently solve the inversion as a minimization problem.
Given the Sea Surface Height (SSH) field of an ocean system, `jaxparrow` estimates the velocity field that best satisfies the cyclogeostrophic balance.

A comprehensive documenation is available: [https://jaxparrow.readthedocs.io/en/latest/](https://jaxparrow.readthedocs.io/en/latest/)!

## Installation

`jaxparrow` is Pip-installable:

```shell
pip install jaxparrow
```

**<ins>However</ins>**, users with access to GPUs or TPUs should first install `JAX` separately in order to fully benefit from its high-performance computing capacities.
See [JAX instructions](https://jax.readthedocs.io/en/latest/installation.html).
By default, `jaxparrow` will install a CPU-only version of JAX if no other version is already present in the Python environment.

## Usage

Estimating the cyclogeostrophic currents from a given Sea Surface Height field can be achieved using any of the following methods:

- [`minimization_based`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.minimization_based),
- [`gradient_wind`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.gradient_wind),
- [`fixed_point`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.fixed_point).

Taking as inputs:

- a SSH field (a `2d jax.Array`),
- the latitude and longitude grids at the T points (two `2d jax.Array`).

They return a result objects holding the cyclogeostrophic velocity $u$ and $v$ components.

In a Python script estimating the cyclogeostrophic currents for a single timestamp would resort to:

```python
from jaxparrow import minimization_based  # or gradient_wind or fixed_point

mb_result = minimization_based(lat_t=lat_2d, lon_t=lon_2d, ssh_t=ssh_2d)

ucg_2d = mb_result.ucg  # 2d jax.Array
vcg_2d = mb_result.vcg  # 2d jax.Array
```

Note that it is also possible to directly pass as inputs the geostrophic velocity $u$ and $v$ components, rather than the SSH:

```python
mb_result = minimization_based(lat_t=lat_2d, lon_t=lon_2d, ug_t=ug_2d, vg_t=vg_2d)
```

To vectorise the estimation of the cyclogeostrophy along a first time dimension, one aims to use `jax.vmap`.

```python
import jax

vmap_cyclogeostrophy = jax.vmap(lambda _ssh_2d: cyclogeostrophy(lat_t=lat_2d, lon_t=lon_2d, ssh_t=_ssh_2d))
mb_result = vmap_cyclogeostrophy(ssh_3d)

ucg_3d = mb_result.ucg  # 3d jax.Array
vcg_3d = mb_result.vcg  # 3d jax.Array
```

`jaxparrow` also notably allows to:

- add arbitrary regularization term when employing the [`minimization-based`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.minimization_based) method, see the [regularization with SWOT data](https://jaxparrow.readthedocs.io/en/latest/examples/swot_regularization/) example notebook,
- reconstruct geostrophic currents, see the [`geostrophy`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.geostrophy.geostrophy) function,
- compute some classical kinematics, see the [`kinematics`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.utils.kinematics) module,
- perform standard operations on the grid, see the [`operators`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.utils.operators) and [`geometry`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.utils.geometry) modules.

Explore `jaxparrow` [documentation](https://jaxparrow.readthedocs.io/en/latest/) for more details, including the [API description](https://jaxparrow.readthedocs.io/en/latest/api/) and step-by-step examples in the form of notebooks.

## Contributing

Contributions are welcomed!
See [CONTRIBUTING.md](https://github.com/meom-group/jaxparrow/blob/main/CONTRIBUTING.md) and [CONDUCT.md](https://github.com/meom-group/jaxparrow/blob/main/CONDUCT.md) to get started.

## How to cite

If you use this software, please cite it: [CITATION.cff](https://github.com/meom-group/jaxparrow/blob/main/CITATION.cff).
Thank you!
