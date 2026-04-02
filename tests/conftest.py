import jax.numpy as jnp
import pytest

import gaussian_eddy
from jaxparrow.utils.geometry import coriolis_factor, GRAVITY


@pytest.fixture
def small_curvilinear_grid():
    """A 10x10 curvilinear grid: lat/lon grid with a small sinusoidal perturbation."""
    lat_1d = jnp.linspace(35.5, 36.5, 10)
    lon_1d = jnp.linspace(-5.0, -4.0, 10)
    lon, lat = jnp.meshgrid(lon_1d, lat_1d)
    # Add a small sinusoidal perturbation to make it curvilinear
    lat = lat + 0.05 * jnp.sin(2 * jnp.pi * lon)
    lon = lon + 0.05 * jnp.sin(2 * jnp.pi * lat)
    land_mask = jnp.zeros_like(lat, dtype=bool)
    return lat, lon, land_mask


@pytest.fixture
def small_grid():
    """A 10x10 lat/lon grid in the Alboran Sea with no land."""
    lat_1d = jnp.linspace(35.5, 36.5, 10)
    lon_1d = jnp.linspace(-5.0, -4.0, 10)
    lon, lat = jnp.meshgrid(lon_1d, lat_1d)
    land_mask = jnp.zeros_like(lat, dtype=bool)
    return lat, lon, land_mask


@pytest.fixture
def gaussian_ssh(small_grid):
    """A simple Gaussian SSH bump on the small grid."""
    lat, lon, _ = small_grid
    R = jnp.sqrt((lat - 36.0) ** 2 + (lon + 4.5) ** 2)
    return -0.2 * jnp.exp(-(R / 0.3) ** 2)


@pytest.fixture
def gaussian_eddy_data():
    """Full Gaussian eddy with analytical velocity fields and derived quantities."""
    R0 = 50e3
    eta0 = -0.2
    latlon0 = (35.92744, -4.03238)

    lat, lon, ssh, ug, vg, ucg, vcg, land_mask = gaussian_eddy.simulate_gaussian_eddy(R0=R0, eta0=eta0)

    R = gaussian_eddy.haversine_distance(lat, latlon0[0], lon, latlon0[1])
    f = coriolis_factor(lat)

    # Interior mask: away from grid edges and eddy boundaries
    interior = (R < R0 * 0.7)
    interior = interior.at[:5, :].set(False).at[-5:, :].set(False)
    interior = interior.at[:, :5].set(False).at[:, -5:].set(False)

    # Annular mask: also away from center (avoids 0/0 in radius of curvature)
    annular = interior & (R > R0 * 0.2)

    return {
        "lat": lat, "lon": lon, "ssh": ssh,
        "ug": ug, "vg": vg, "ucg": ucg, "vcg": vcg,
        "land_mask": land_mask,
        "R0": R0, "eta0": eta0, "R": R, "f": f,
        "interior": interior, "annular": annular,
    }
