import jax
import jax.numpy as jnp
from jaxtyping import Float

from ._core import (
    CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result
)
from ..utils import kinematics


def gradient_wind(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    is_grid_rectilinear: bool | None = None,
    rotate_to_geographic: bool = True,
    return_geos: bool = False
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using the gradient wind approximation.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``land_mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``land_mask``). Geostrophic velocities are provided on the T grid.

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitude of the T grid
    lon_t : Float[jax.Array, "y x"]
        Longitude of the T grid
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field (on the T grid)

        Defaults to `None`, required if geostrophic velocities are not provided
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` or ``ug_t`` `nan` values

        Defaults to `None`
    is_grid_rectilinear : bool, optional
        If `True`, the grid is assumed to be rectilinear in geographic coordinates.
        If `False`, the grid is assumed to be curvilinear and the grid angle is computed from the grid spacing. 
        If `None`, the grid is assumed to be rectilinear if the grid angle computed from the grid spacing is close to zero everywhere, and curvilinear otherwise.

        Defaults to `None`
    rotate_to_geographic : bool, optional
        If `True`, rotates the output velocities from grid-relative to geographic coordinates.
        Rotation is performed using the grid angle computed from the grid spacing.
        If `False`, output velocities are in grid-relative coordinates.

        If using a rectilinear grid in geographic coordinates, set to `False` to avoid unnecessary rotation.

        Defaults to `True`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`

    Returns
    -------
    CyclogeostrophyResult
        Named tuple containing:
        - ``ucg``: $u$ component of cyclogeostrophic velocity, on the T grid
        - ``vcg``: $v$ component of cyclogeostrophic velocity, on the T grid
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask, is_grid_rectilinear=is_grid_rectilinear
    )

    ucg, vcg = _gradient_wind(
        setup.ug_t, setup.vg_t,
        setup.dx_t, setup.dy_t, 
        setup.coriolis_factor_t, 
        setup.land_mask
    )

    return assemble_result(ucg, vcg, setup, rotate_to_geographic, return_geos)


@jax.jit
def _gradient_wind(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    R = kinematics._radius_of_curvature(ug_t, vg_t, dx_t, dy_t, land_mask)

    V_g = kinematics.magnitude(ug_t, vg_t, land_mask, uv_on_t=True)
    V_gr = 2 * V_g / (1 + jnp.sqrt(1 + 4 * V_g / (coriolis_factor_t * R)))

    ratio = V_gr / V_g

    ucg = ratio * ug_t
    vcg = ratio * vg_t

    return ucg, vcg
