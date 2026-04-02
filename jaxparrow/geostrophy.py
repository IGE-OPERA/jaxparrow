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
    land_mask: Float[jax.Array, "y x"] = None,
    is_grid_rectilinear: bool | None = None,
    rotate_to_geographic: bool = True
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
    is_grid_rectilinear : bool, optional
        If `True`, the grid is assumed to be rectilinear in geographic coordinates.
        If `False`, the grid is assumed to be curvilinear and the grid angle is computed from the grid spacing. 
        If `None`, the grid is assumed to be rectilinear if the grid angle computed from the grid spacing is close to zero everywhere, and curvilinear otherwise.

        Defaults to `None`
    rotate_to_geographic : bool, optional
        If `True`, rotates the velocity field to geographic coordinates (eastward and northward components).

        Defaults to `True`, in which case the returned velocity components are in geographic coordinates. 
        If `False`, the returned velocity components are in grid coordinates (i.e. along the grid axes, which may not be aligned with geographic east and north directions).

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

    if rotate_to_geographic:
        grid_angle = None
        if is_grid_rectilinear is None:
            # determine if the grid is rectilinear by checking the grid angle
            grid_angle = geometry.compute_grid_angle(lat_t, lon_t)
            is_grid_rectilinear = jnp.all(jnp.abs(grid_angle) < 1e-3)
        
        if not is_grid_rectilinear:
            if grid_angle is None:
                grid_angle = geometry.compute_grid_angle(lat_t, lon_t)
            ug_t, vg_t = geometry.rotate_to_geographic(ug_t, vg_t, grid_angle)

    return ug_t, vg_t


@jax.jit
def _geostrophy(
    ssh_t: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    deta_x_t, deta_y_t = operators.horizontal_derivatives(ssh_t, lat=lat_t, lon=lon_t, land_mask=land_mask)

    f_t = geometry.coriolis_factor(lat_t)

    # Computing the geostrophic velocities
    # u = -g/f * dη/dy
    # v =  g/f * dη/dx
    ug_t = -geometry.GRAVITY * deta_y_t / f_t
    vg_t = geometry.GRAVITY * deta_x_t / f_t

    return ug_t, vg_t
