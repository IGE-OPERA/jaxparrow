from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float

from ..geostrophy import geostrophy
from ..utils import geometry, operators, sanitize


# =============================================================================
# Types
# =============================================================================

class CyclogeostrophySetup(NamedTuple):
    lat_t: Float[jax.Array, "y x"]
    lon_t: Float[jax.Array, "y x"]
    land_mask: Float[jax.Array, "y x"]
    ug_t: Float[jax.Array, "y x"]
    vg_t: Float[jax.Array, "y x"]
    dx_t: Float[jax.Array, "y x"]
    dy_t: Float[jax.Array, "y x"]
    coriolis_factor_t: Float[jax.Array, "y x"]
    is_grid_rectilinear: bool
    grid_angle: Float[jax.Array, "y x"]


class CyclogeostrophyResult(NamedTuple):
    """
    Result of cyclogeostrophic velocity computation.

    This NamedTuple provides named access to results, avoiding positional unpacking errors.
    All fields except ``ucg`` and ``vcg`` are optional and depend on the
    ``return_*`` flags passed to the computation function.

    Attributes
    ----------
    ucg : Float[jax.Array, "y x"]
        $u$ component of cyclogeostrophic velocity, on the T grid
    vcg : Float[jax.Array, "y x"]
        $v$ component of cyclogeostrophic velocity, on the T grid
    ug : Float[jax.Array, "y x"] | None
        $u$ component of geostrophic velocity, on the T grid (if ``return_geos=True``)
    vg : Float[jax.Array, "y x"] | None
        $v$ component of geostrophic velocity, on the T grid (if ``return_geos=True``)
    losses : Float[jax.Array, "n_it"] | None
        Cyclogeostrophic imbalance over iterations (if ``return_losses=True``)
    """

    ucg: Float[jax.Array, "y x"]
    vcg: Float[jax.Array, "y x"]
    ug: Float[jax.Array, "y x"] = None
    vg: Float[jax.Array, "y x"] = None
    losses: Float[jax.Array, "n_it"] = None


# =============================================================================
# Setup and Result Assembly
# =============================================================================

def setup_cyclogeostrophy(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    is_grid_rectilinear: bool | None = None,
) -> CyclogeostrophySetup:
    # Check if geostrophic velocities are provided directly
    use_geos_directly = ug_t is not None and vg_t is not None

    grid_angle = geometry.compute_grid_angle(lat_t, lon_t)
    if is_grid_rectilinear is None:
        # determine if the grid is rectilinear by checking the grid angle
        is_grid_rectilinear = jnp.all(jnp.abs(grid_angle) < 1e-3)

    if use_geos_directly:
        land_mask = sanitize.init_land_mask(ug_t, land_mask)

        if not is_grid_rectilinear:
            # rotate the input velocities to the grid coordinates
            ug_t, vg_t = geometry.rotate_to_grid(ug_t, vg_t, grid_angle)
    else:
        # SSH-based computation
        if ssh_t is None:
            raise ValueError(
                "Either provide ssh_t to compute geostrophic velocities from SSH, "
                "or provide ug_t, vg_t directly on the T grid."
            )
        
        land_mask = sanitize.init_land_mask(ssh_t, land_mask)
        ug_t, vg_t = geostrophy(ssh_t, lat_t, lon_t, land_mask, rotate_to_geographic=False)

    dx, dy = geometry.grid_spacing(lat_t, lon_t)
    f = geometry.coriolis_factor(lat_t)

    return CyclogeostrophySetup(
        lat_t=lat_t, lon_t=lon_t, 
        land_mask=land_mask, 
        ug_t=ug_t, vg_t=vg_t, 
        dx_t=dx, dy_t=dy, 
        coriolis_factor_t=f, 
        is_grid_rectilinear=is_grid_rectilinear,
        grid_angle=grid_angle
    )


def assemble_result(
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    setup: CyclogeostrophySetup,
    rotate_to_geographic: bool,
    return_geos: bool,
    return_losses: bool = False,
    losses: Float[jax.Array, "n_it"] = None,
) -> CyclogeostrophyResult:
    # Handle masked data (set land cells to NaN)
    ucg_t = sanitize.sanitize_data(ucg_t, jnp.nan, setup.land_mask)
    vcg_t = sanitize.sanitize_data(vcg_t, jnp.nan, setup.land_mask)

    if return_geos:
        ug_out = setup.ug_t
        vg_out = setup.vg_t
    else:
        ug_out = None
        vg_out = None

    if rotate_to_geographic:
        if not setup.is_grid_rectilinear:
            ucg_t, vcg_t = geometry.rotate_to_geographic(ucg_t, vcg_t, setup.grid_angle)
            if ug_out is not None and vg_out is not None:
                ug_out, vg_out = geometry.rotate_to_geographic(ug_out, vg_out, setup.grid_angle)

    return CyclogeostrophyResult(
        ucg=ucg_t,
        vcg=vcg_t,
        ug=ug_out,
        vg=vg_out,
        losses=losses if return_losses else None,
    )


# =============================================================================
# Public API Functions
# =============================================================================

def cyclogeostrophic_loss(
    ug: Float[jax.Array, "y x"],
    vg: Float[jax.Array, "y x"],
    ucg: Float[jax.Array, "y x"],
    vcg: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True
) -> Float[jax.Array, ""]:
    """
    Computes the cyclogeostrophic imbalance loss (a scalar) from a geostrophic and a cyclogeostrophic velocity field.

    The velocity fields can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    ug : Float[jax.Array, "y x"]
        $u$ component of the geostrophic velocity field
    vg : Float[jax.Array, "y x"]
        $v$ component of the geostrophic velocity field
    ucg : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic velocity field
    vcg : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic velocity field
    lat_t : Float[jax.Array, "y x"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[jax.Array, "y x"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[jax.Array, "y x"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[jax.Array, "y x"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[jax.Array, "y x"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[jax.Array, "y x"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the T grid 
        (this is important when manipulating staggered grids)
        
        Defaults to `True`
    Returns
    -------
    loss : Float[jax.Array, ""]
        Cyclogeostrophic imbalance loss
    """
    u_imbalance, v_imbalance = cyclogeostrophic_imbalance(
        ug, vg, ucg, vcg, lat_t, lon_t, lat_u, lon_u, lat_v, lon_v, land_mask, uv_on_t
    )

    return jnp.nansum(u_imbalance ** 2 + v_imbalance ** 2)


def cyclogeostrophic_imbalance(
    ug: Float[jax.Array, "y x"],
    vg: Float[jax.Array, "y x"],
    ucg: Float[jax.Array, "y x"],
    vcg: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Computes the cyclogeostrophic imbalance field from a geostrophic and a cyclogeostrophic velocity field.

    The velocity fields can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    ug : Float[jax.Array, "y x"]
        $u$ component of the geostrophic velocity field
    vg : Float[jax.Array, "y x"]
        $v$ component of the geostrophic velocity field
    ucg : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic velocity field
    vcg : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic velocity field
    lat_t : Float[jax.Array, "y x"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[jax.Array, "y x"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[jax.Array, "y x"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[jax.Array, "y x"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[jax.Array, "y x"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[jax.Array, "y x"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the T grid 
        (this is important when manipulating staggered grids)
        
        Defaults to `True`

    Returns
    -------
    u_imbalance : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic imbalance, on the T grid
    v_imbalance : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic imbalance, on the T grid
    """
    if land_mask is None:
        land_mask = sanitize.init_land_mask(ug)

    if not uv_on_t:
        ug = operators.interpolation(ug, axis=1, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        vg = operators.interpolation(vg, axis=0, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        ucg = operators.interpolation(ucg, axis=1, padding="right", land_mask=land_mask)
        vcg = operators.interpolation(vcg, axis=0, padding="right", land_mask=land_mask)

    if lat_t is None or lon_t is None:
        if lat_u is not None and lon_u is not None:
            lat_t = operators.interpolation(lat_u, axis=1, padding="left", land_mask=land_mask)
            lon_t = operators.interpolation(lon_u, axis=1, padding="left", land_mask=land_mask)
        elif lat_v is not None and lon_v is not None:
            lat_t = operators.interpolation(lat_v, axis=0, padding="left", land_mask=land_mask)
            lon_t = operators.interpolation(lon_v, axis=0, padding="left", land_mask=land_mask)
        else:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
    
    # compute grid spacing once
    dx, dy = geometry.grid_spacing(lat_t, lon_t)
    f = geometry.coriolis_factor(lat_t)

    return _cyclogeostrophic_imbalance(ug, vg, ucg, vcg, dx, dy, f, land_mask)


# =============================================================================
# Internal Functions
# =============================================================================


def _cyclogeostrophic_loss(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, ""]:
    u_imbalance, v_imbalance = _cyclogeostrophic_imbalance(
        ug_t, vg_t, ucg_t, vcg_t, dx_t, dy_t, coriolis_factor_t, land_mask
    )

    return jnp.nansum(u_imbalance ** 2 + v_imbalance ** 2)


def _cyclogeostrophic_imbalance(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    u_adv_t, v_adv_t = _advection(ucg_t, vcg_t, dx_t, dy_t, land_mask)

    u_imbalance = ucg_t + v_adv_t / coriolis_factor_t - ug_t
    v_imbalance = vcg_t - u_adv_t / coriolis_factor_t - vg_t

    return u_imbalance, v_imbalance


def _advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    u_adv = _u_advection(u_t, v_t, dx_t, dy_t, land_mask)
    v_adv = _v_advection(u_t, v_t, dx_t, dy_t, land_mask)

    return u_adv, v_adv


def _u_advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, "y x"]:
    du_x_t, du_y_t = operators.horizontal_derivatives(u_t, dx=dx_t, dy=dy_t, land_mask=land_mask)

    u_adv = u_t * du_x_t + v_t * du_y_t

    return u_adv


def _v_advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, "y x"]:
    dv_x_t, dv_y_t = operators.horizontal_derivatives(v_t, dx=dx_t, dy=dy_t, land_mask=land_mask)

    v_adv = u_t * dv_x_t + v_t * dv_y_t

    return v_adv
