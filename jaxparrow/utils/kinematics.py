import jax
import jax.numpy as jnp
from jaxtyping import Float

from .geometry import coriolis_factor, grid_spacing
from .operators import horizontal_derivatives, interpolation
from .sanitize import init_land_mask, sanitize_data


def setup_kinematics(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
):
    land_mask = init_land_mask(u, land_mask)

    if not uv_on_t:
        u = interpolation(u, axis=1, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        v = interpolation(v, axis=0, padding="left", land_mask=land_mask)  # V(j), V(j+1) -> T(j+1)

    if lat_t is None or lon_t is None:
        if lat_u is not None and lon_u is not None:
            lat_t = interpolation(lat_u, axis=1, padding="left", land_mask=land_mask)
            lon_t = interpolation(lon_u, axis=1, padding="left", land_mask=land_mask)
        elif lat_v is not None and lon_v is not None:
            lat_t = interpolation(lat_v, axis=0, padding="left", land_mask=land_mask)
            lon_t = interpolation(lon_v, axis=0, padding="left", land_mask=land_mask)
        else:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
    
    # compute grid metrics once
    dx, dy = grid_spacing(lat_t, lon_t)

    return u, v, lat_t, lon_t, dx, dy, land_mask


def magnitude(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True
) -> Float[jax.Array, "y x"]:
    """
    Computes the magnitude (azimuthal velocity) of a 2d velocity field.

    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        $u$ component of the velocity field (on the U or T grid)
    v : Float[jax.Array, "y x"]
        $v$ component of the velocity field (on the V or T grid)
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        Defaults to `None`
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.
        
        Defaults to `True`

    Returns
    -------
    magnitude : Float[jax.Array, "y x"]
        Magnitude of the velocity field, on the T grid
    """
    land_mask = init_land_mask(u, land_mask)

    if not uv_on_t:
        u = interpolation(u, axis=1, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        v = interpolation(v, axis=0, padding="left", land_mask=land_mask)  # V(j), V(j+1) -> T(j+1)

    magn = jnp.sqrt(u** 2 + v ** 2)
    magn = sanitize_data(magn, jnp.nan, land_mask)

    return magn


def kinetic_energy(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
) -> Float[jax.Array, "y x"]:
    """
    Computes the Kinetic Energy (KE) of a velocity field.
    
    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        $u$ component of the velocity field (on the U grid)
    v : Float[jax.Array, "y x"]
        $v$ component of the velocity field (on the V grid)
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).

        Defaults to `None`
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.

        Defaults to `True`

    Returns
    -------
    kinetic_energy : Float[jax.Array, "y x"]
        The Kinetic Energy on the T grid
    """
    # Make sure the mask is initialized
    land_mask = init_land_mask(u, land_mask)
    
    if not uv_on_t:
        u = interpolation(u, axis=1, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        v = interpolation(v, axis=0, padding="left", land_mask=land_mask)  # V(j), V(j+1) -> T(j+1)

    ke = (u ** 2 + v ** 2) / 2
    ke = sanitize_data(ke, jnp.nan, land_mask)

    return ke


def vorticity(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
    normalize_by_coriolis: bool = True
) -> Float[jax.Array, "y x"]:
    """
    Computes the relative vorticity of a velocity field.

    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        $u$ component of the velocity field
    v : Float[jax.Array, "y x"]
        $v$ component of the velocity field
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
    normalize_by_coriolis : bool, optional
        If `True`, returns the vorticity normalized by the Coriolis factor

        Defaults to `True`

    Returns
    -------
    vorticity : Float[jax.Array, "y x"]
        The vorticity on the T grid, normalized by the Coriolis factor if ``normalize_by_coriolis=True``
    """
    u, v, lat_t, lon_t, dx, dy, land_mask = setup_kinematics(
        u, v, lat_t, lon_t, lat_u, lon_u, lat_v, lon_v, land_mask, uv_on_t
    )

    _, du_y = horizontal_derivatives(u, dx=dx, dy=dy, land_mask=land_mask)
    dv_x, _ = horizontal_derivatives(v, dx=dx, dy=dy, land_mask=land_mask)

    vort = dv_x - du_y

    if normalize_by_coriolis:
        f = coriolis_factor(lat_t)
        vort /= f

    vort = sanitize_data(vort, jnp.nan, land_mask)

    return vort


def strain_rate(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
    normalize_by_coriolis: bool = True
) -> Float[jax.Array, "y x"]:
    """
    Computes the strain rate magnitude of a velocity field.

    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        $u$ component of the velocity field
    v : Float[jax.Array, "y x"]
        $v$ component of the velocity field
    lat_t : Float[jax.Array, "y x"], optional
        Latitudes of the T grid.
        Defaults to `None`
    lon_t : Float[jax.Array, "y x"], optional
        Longitudes of the T grid.
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
    normalize_by_coriolis : bool, optional
        If `True`, returns the strain rate normalized by the Coriolis factor

        Defaults to `True`

    Returns
    -------
    strain_rate : Float[jax.Array, "y x"]
        The strain rate magnitude on the T grid, normalized by the Coriolis factor if ``normalize_by_coriolis=True``
    """
    u, v, lat_t, lon_t, dx, dy, land_mask = setup_kinematics(
        u, v, lat_t, lon_t, lat_u, lon_u, lat_v, lon_v, land_mask, uv_on_t
    )

    du_x, du_y = horizontal_derivatives(u, dx=dx, dy=dy, land_mask=land_mask)
    dv_x, dv_y = horizontal_derivatives(v, dx=dx, dy=dy, land_mask=land_mask)

    strain = jnp.sqrt((du_x - dv_y) ** 2 + (dv_x + du_y) ** 2)

    if normalize_by_coriolis:
        f = coriolis_factor(lat_t)
        strain /= f

    strain = sanitize_data(strain, jnp.nan, land_mask)

    return strain


def radius_of_curvature(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
) -> Float[jax.Array, "y x"]:
    """
    Computes the radius of curvature of a 2d velocity field.

    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        $u$ component of the velocity field
    v : Float[jax.Array, "y x"]
        $v$ component of the velocity field
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
    rc : Float[jax.Array, "y x"]
        The radius of curvature of the velocity field in meters, on the T grid
    """
    u, v, lat_t, lon_t, dx, dy, land_mask = setup_kinematics(
        u, v, lat_t, lon_t, lat_u, lon_u, lat_v, lon_v, land_mask, uv_on_t
    )

    return _radius_of_curvature(u, v, dx, dy, land_mask)


def _radius_of_curvature(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"]
) -> Float[jax.Array, "y x"]:
    V_t = magnitude(u_t, v_t, land_mask, uv_on_t=True)

    du_x_t, du_y_t = horizontal_derivatives(u_t, dx=dx_t, dy=dy_t, land_mask=land_mask)
    dv_x_t, dv_y_t = horizontal_derivatives(v_t, dx=dx_t, dy=dy_t, land_mask=land_mask)

    numerator = V_t ** 3
    denominator = u_t ** 2 * dv_x_t - v_t ** 2 * du_y_t - u_t * v_t * (du_x_t - dv_y_t)
    r = numerator / denominator

    return r
