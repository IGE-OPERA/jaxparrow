import jax
import jax.numpy as jnp
from jaxtyping import Float

from .operators import interpolation


#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81


def grid_spacing(
    lat: Float[jax.Array, "y x"], lon: Float[jax.Array, "y x"]
) -> tuple[
    Float[jax.Array, "y x"], 
    Float[jax.Array, "y x"], 
]:
    """
    Computes the physical spacing associated with one grid-index step, 
    used to transform derivatives to physical coordinates.

    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of `cos` and `arccos`
    functions to avoid truncation issues.

    Parameters
    ----------
    lat : Float[jax.Array, "y x"]
        Latitude grid
    lon : Float[jax.Array, "y x"]
        Longitude grid

    Returns
    -------
    dx : Float[jax.Array, "y x"]
        Spacing associated with one step in the x-index direction
    dy : Float[jax.Array, "y x"]
        Spacing associated with one step in the y-index direction
    """
    def physical_spacing(lat1, lat2, lon1, lon2):
        # convert to radians
        lat1_rad = jnp.radians(lat1)
        lat2_rad = jnp.radians(lat2)

        # difference in radians; normalize lon diff to [-180, 180] before radians to handle dateline
        dlon = lon2 - lon1
        dlon = (dlon + 180.0) % 360.0 - 180.0   # now in [-180,180]
        dlon_rad = jnp.radians(dlon)

        dlat_rad = jnp.radians(lat2 - lat1)

        # haversine distance
        a = jnp.sin(dlat_rad / 2.0) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * (jnp.sin(dlon_rad / 2.0) ** 2)
        c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
        d = EARTH_RADIUS * c

        return d

    # physical spacing
    dx = physical_spacing(lat[:, :-1], lat[:, 1:], lon[:, :-1], lon[:, 1:])
    dy = physical_spacing(lat[:-1, :], lat[1:, :], lon[:-1, :], lon[1:, :])

    dx = jnp.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy = jnp.pad(dy, ((0, 1), (0, 0)), mode="edge")

    return dx, dy


def _axis_bearing_to_angle(
    lat: Float[jax.Array, "lat lon"],
    lon: Float[jax.Array, "lat lon"],
    axis: int
) -> Float[jax.Array, "lat lon"]:
    """Compute the angle (counterclockwise from east) of a grid axis direction."""
    lat_rad = jnp.radians(lat)

    if axis == 1:
        # differences along axis=1 (i-direction, columns)
        dlon = jnp.zeros_like(lon)
        dlon = dlon.at[:, 1:-1].set(lon[:, 2:] - lon[:, :-2])
        dlon = dlon.at[:, 0].set(lon[:, 1] - lon[:, 0])
        dlon = dlon.at[:, -1].set(lon[:, -1] - lon[:, -2])

        lat1_rad = jnp.zeros_like(lat_rad)
        lat1_rad = lat1_rad.at[:, 1:-1].set(lat_rad[:, :-2])
        lat1_rad = lat1_rad.at[:, 0].set(lat_rad[:, 0])
        lat1_rad = lat1_rad.at[:, -1].set(lat_rad[:, -2])

        lat2_rad = jnp.zeros_like(lat_rad)
        lat2_rad = lat2_rad.at[:, 1:-1].set(lat_rad[:, 2:])
        lat2_rad = lat2_rad.at[:, 0].set(lat_rad[:, 1])
        lat2_rad = lat2_rad.at[:, -1].set(lat_rad[:, -1])
    else:
        # differences along axis=0 (j-direction, rows)
        dlon = jnp.zeros_like(lon)
        dlon = dlon.at[1:-1, :].set(lon[2:, :] - lon[:-2, :])
        dlon = dlon.at[0, :].set(lon[1, :] - lon[0, :])
        dlon = dlon.at[-1, :].set(lon[-1, :] - lon[-2, :])

        lat1_rad = jnp.zeros_like(lat_rad)
        lat1_rad = lat1_rad.at[1:-1, :].set(lat_rad[:-2, :])
        lat1_rad = lat1_rad.at[0, :].set(lat_rad[0, :])
        lat1_rad = lat1_rad.at[-1, :].set(lat_rad[-2, :])

        lat2_rad = jnp.zeros_like(lat_rad)
        lat2_rad = lat2_rad.at[1:-1, :].set(lat_rad[2:, :])
        lat2_rad = lat2_rad.at[0, :].set(lat_rad[1, :])
        lat2_rad = lat2_rad.at[-1, :].set(lat_rad[-1, :])

    dlon = (dlon + 180.0) % 360.0 - 180.0
    dlon_rad = jnp.radians(dlon)

    x = jnp.sin(dlon_rad) * jnp.cos(lat2_rad)
    y = jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(lat2_rad) * jnp.cos(dlon_rad)
    bearing = jnp.arctan2(x, y)  # clockwise from north

    # Convert to angle counterclockwise from east, wrapped to [-pi, pi]
    angle = jnp.pi / 2 - bearing
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def compute_grid_angle(
    lat: Float[jax.Array, "lat lon"],
    lon: Float[jax.Array, "lat lon"]
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Computes the local angles of both grid axes relative to geographic east.

    For curvilinear grids (e.g., SWOT swaths, tripolar grids), the grid axes are not aligned
    with geographic east-west/north-south directions. This function computes the rotation angles
    needed to transform velocity components between grid coordinates and geographic coordinates.

    Parameters
    ----------
    lat : Float[jax.Array, "lat lon"]
        Latitude grid
    lon : Float[jax.Array, "lat lon"]
        Longitude grid

    Returns
    -------
    angle_i : Float[jax.Array, "lat lon"]
        Angle of the grid i-axis (axis=1) relative to geographic east, in radians,
        measured counterclockwise. Range is [-pi, pi].
    angle_j : Float[jax.Array, "lat lon"]
        Angle of the grid j-axis (axis=0) relative to geographic east, in radians,
        measured counterclockwise. Range is [-pi, pi].

    Notes
    -----
    Both angles are computed using the initial bearing formula between adjacent grid points.
    For a standard rectilinear lat/lon grid: ``angle_i ≈ 0`` (i-axis ≈ east) and
    ``angle_j ≈ π/2`` (j-axis ≈ north).

    For ascending SWOT passes (satellite heading north), ``angle_j > 0``.
    For descending SWOT passes (satellite heading south, rows increasing southward),
    ``angle_j < 0``, making the grid left-handed. The rotation functions
    :func:`rotate_to_geographic` and :func:`rotate_to_grid` handle both cases correctly.
    """
    angle_i = _axis_bearing_to_angle(lat, lon, axis=1)
    angle_j = _axis_bearing_to_angle(lat, lon, axis=0)
    return angle_i, angle_j


def rotate_to_geographic(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    angle_i: Float[jax.Array, "y x"],
    angle_j: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Rotates velocity components from grid coordinates to geographic coordinates (eastward and northward components).

    Uses the full 2-column rotation matrix defined by the actual directions of both grid axes,
    which correctly handles right-handed grids (ascending passes) and left-handed grids
    (descending passes where rows increase southward).

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        Velocity component along the grid i-axis (axis=1)
    v : Float[jax.Array, "y x"]
        Velocity component along the grid j-axis (axis=0)
    angle_i : Float[jax.Array, "y x"]
        Angle of the grid i-axis (axis=1) relative to geographic east, in radians
        (counterclockwise positive). Typically obtained from :func:`compute_grid_angle`.
    angle_j : Float[jax.Array, "y x"]
        Angle of the grid j-axis (axis=0) relative to geographic east, in radians
        (counterclockwise positive). Typically obtained from :func:`compute_grid_angle`.

    Returns
    -------
    u_east : Float[jax.Array, "y x"]
        Eastward velocity component
    v_north : Float[jax.Array, "y x"]
        Northward velocity component
    """
    cos_i = jnp.cos(angle_i)
    sin_i = jnp.sin(angle_i)
    cos_j = jnp.cos(angle_j)
    sin_j = jnp.sin(angle_j)

    # det = sin(angle_j - angle_i): +1 for right-handed (ascending), -1 for left-handed (descending)
    det = cos_i * sin_j - sin_i * cos_j

    u_east = (u * cos_i + v * cos_j) / det
    v_north = (u * sin_i + v * sin_j) / det

    return u_east, v_north


def rotate_to_grid(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    angle_i: Float[jax.Array, "y x"],
    angle_j: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Rotates velocity components from geographic coordinates (eastward and northward) to grid coordinates.

    This is the inverse of :func:`rotate_to_geographic`.

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        Eastward velocity component
    v : Float[jax.Array, "y x"]
        Northward velocity component
    angle_i : Float[jax.Array, "y x"]
        Angle of the grid i-axis (axis=1) relative to geographic east, in radians
        (counterclockwise positive). Typically obtained from :func:`compute_grid_angle`.
    angle_j : Float[jax.Array, "y x"]
        Angle of the grid j-axis (axis=0) relative to geographic east, in radians
        (counterclockwise positive). Typically obtained from :func:`compute_grid_angle`.

    Returns
    -------
    u_grid : Float[jax.Array, "y x"]
        Velocity component along the grid i-axis (axis=1)
    v_grid : Float[jax.Array, "y x"]
        Velocity component along the grid j-axis (axis=0)
    """
    cos_i = jnp.cos(angle_i)
    sin_i = jnp.sin(angle_i)
    cos_j = jnp.cos(angle_j)
    sin_j = jnp.sin(angle_j)

    u_grid = u * sin_j - v * cos_j
    v_grid = -u * sin_i + v * cos_i

    return u_grid, v_grid


def coriolis_factor(lat: Float[jax.Array, "y x"]) -> Float[jax.Array, "y x"]:
    """
    Computes the Coriolis factor from a latitude grid.

    Parameters
    ----------
    lat : Float[jax.Array, "y x"]
        Latitudes grid

    Returns
    -------
    cf : Float[jax.Array, "y x"]
        Coriolis factor grid
    """
    return 2 * EARTH_ANG_SPEED * jnp.sin((jnp.radians(lat)))


def compute_uv_grids(
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"]
) -> tuple[
    Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]
]:
    """
    Computes the U and V grids associated to a T grid following NEMO convention.

    Parameters
    ----------
    lat_t : Float[jax.Array, "lat lon"]
        Latitudes of the T grid
    lon_t : Float[jax.Array, "lat lon"]
        Longitudes of the T grid

    Returns
    -------
    lat_u : Float[jax.Array, "lat lon"]
        Latitudes of the U grid
    lon_u : Float[jax.Array, "lat lon"]
        Longitudes of the U grid
    lat_v : Float[jax.Array, "lat lon"]
        Latitudes of the V grid
    lon_v : Float[jax.Array, "lat lon"]
        Longitudes of the V grid
    """
    lat_u = interpolation(lat_t, axis=1, padding="right")
    lat_u = lat_u.at[:, -1].set(2 * lat_t[:, -1] - lat_t[:, -2])
    lon_u = interpolation(lon_t, axis=1, padding="right")
    lon_u = lon_u.at[:, -1].set(2 * lon_t[:, -1] - lon_t[:, -2])

    lat_v = interpolation(lat_t, axis=0, padding="right")
    lat_v = lat_v.at[-1, :].set(2 * lat_t[-1, :] - lat_t[-2, :])
    lon_v = interpolation(lon_t, axis=0, padding="right")
    lon_v = lon_v.at[-1, :].set(2 * lon_t[-1, :] - lon_t[-2, :])

    return lat_u, lon_u, lat_v, lon_v
