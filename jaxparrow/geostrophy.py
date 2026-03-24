import jax
import jax.numpy as jnp
from jaxtyping import Float

from .utils import geometry, operators, sanitize


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(
    ssh_t: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"] = None
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Computes the geostrophic velocity field from a Sea Surface Height (SSH) field.

    Parameters
    ----------
    ssh_t : Float[jax.Array, "y x"]
        SSH field (on the T grid)
    lat_t : Float[jax.Array, "y x"]
        Latitudes of the T grid
    lon_t : Float[jax.Array, "y x"]
        Longitudes of the T grid
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).

        Defaults to `None`, in which case inferred from `ssh_t` `nan` values

    Returns
    -------
    ug_t : Float[jax.Array, "y x"]
        $u$ component of the geostrophic velocity field, on the T grid
    vg_t : Float[jax.Array, "y x"]
        $v$ component of the geostrophic velocity field, on the T grid
    """
    # Make sure the mask is initialized
    land_mask = sanitize.init_land_mask(ssh_t, land_mask)

    # Handle spurious and masked data
    ssh_t = sanitize.sanitize_data(ssh_t, jnp.nan, land_mask)

    ug_t, vg_t = _geostrophy(ssh_t, lat_t, lon_t, land_mask)

    # Handle masked data (set land cells to NaN)
    ug_t = sanitize.sanitize_data(ug_t, jnp.nan, land_mask)
    vg_t = sanitize.sanitize_data(vg_t, jnp.nan, land_mask)

    return ug_t, vg_t


@jax.jit
def _geostrophy(
    ssh_t: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    deta_e_t, deta_n_t = operators.horizontal_derivatives(ssh_t, lat=lat_t, lon=lon_t, land_mask=land_mask)

    f_t = geometry.coriolis_factor(lat_t)

    # Computing the geostrophic velocities
    # u = -g/f * dη/dn  (eastward velocity)
    # v =  g/f * dη/de  (northward velocity)
    ug_t = -geometry.GRAVITY * deta_n_t / f_t
    vg_t = geometry.GRAVITY * deta_e_t / f_t

    return ug_t, vg_t
