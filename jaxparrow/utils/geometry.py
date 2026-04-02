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


def compute_grid_angle(
    lat: Float[jax.Array, "lat lon"],
    lon: Float[jax.Array, "lat lon"]
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the local angle of the grid i-axis (axis=1) relative to geographic east.

    For curvilinear grids (e.g., SWOT swaths, tripolar grids), the grid axes are not aligned
    with geographic east-west/north-south directions. This function computes the rotation angle
    needed to transform gradients from grid coordinates to geographic coordinates.

    The angle is measured counterclockwise from geographic east to the grid i-direction.

    Parameters
    ----------
    lat : Float[jax.Array, "lat lon"]
        Latitude grid
    lon : Float[jax.Array, "lat lon"]
        Longitude grid

    Returns
    -------
    angle : Float[jax.Array, "lat lon"]
        Rotation angle in radians, measured counterclockwise from geographic east
        to the grid i-direction. Range is [-pi, pi].

    Notes
    -----
    The angle is computed using the initial bearing formula between adjacent grid points
    along the i-axis (axis=1). The formula computes the azimuth from north (clockwise positive),
    which is then converted to angle from east (counterclockwise positive).

    For orthogonal grids, the j-axis direction is at angle + pi/2.
    """
    # Use central differences where possible, forward/backward at boundaries
    lat_rad = jnp.radians(lat)

    # Compute differences in longitude (handling wraparound)
    dlon = jnp.zeros_like(lon)
    dlon = dlon.at[:, 1:-1].set(lon[:, 2:] - lon[:, :-2])  # central diff
    dlon = dlon.at[:, 0].set(lon[:, 1] - lon[:, 0])  # forward diff at left
    dlon = dlon.at[:, -1].set(lon[:, -1] - lon[:, -2])  # backward diff at right

    # Normalize to [-180, 180]
    dlon = (dlon + 180.0) % 360.0 - 180.0
    dlon_rad = jnp.radians(dlon)

    # Compute latitude at neighboring points for bearing calculation
    lat1_rad = jnp.zeros_like(lat_rad)
    lat1_rad = lat1_rad.at[:, 1:-1].set(lat_rad[:, :-2])
    lat1_rad = lat1_rad.at[:, 0].set(lat_rad[:, 0])
    lat1_rad = lat1_rad.at[:, -1].set(lat_rad[:, -2])

    lat2_rad = jnp.zeros_like(lat_rad)
    lat2_rad = lat2_rad.at[:, 1:-1].set(lat_rad[:, 2:])
    lat2_rad = lat2_rad.at[:, 0].set(lat_rad[:, 1])
    lat2_rad = lat2_rad.at[:, -1].set(lat_rad[:, -1])

    # Initial bearing formula: bearing from point 1 to point 2
    # bearing = atan2(sin(dlon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
    # This gives bearing measured clockwise from north
    x = jnp.sin(dlon_rad) * jnp.cos(lat2_rad)
    y = jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(lat2_rad) * jnp.cos(dlon_rad)
    bearing = jnp.arctan2(x, y)  # radians, clockwise from north

    # Convert bearing (clockwise from north) to angle (counterclockwise from east)
    # If bearing = 0 (north), angle = pi/2
    # If bearing = pi/2 (east), angle = 0
    # angle = pi/2 - bearing
    angle = jnp.pi / 2 - bearing

    return angle


def rotate_to_geographic(
    u: Float[jax.Array, "y x"], 
    v: Float[jax.Array, "y x"], 
    grid_angle: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Rotates velocity components from grid coordinates to geographic coordinates (eastward and northward components).

    Parameters
    ----------
    u : Float[jax.Array, "y x"]
        Velocity component along the grid x-axis
    v : Float[jax.Array, "y x"]
        Velocity component along the grid y-axis
    grid_angle : Float[jax.Array, "y x"]
        Angle between the grid x-axis and the eastward direction, in radians. Positive values indicate a counter-clockwise rotation from the grid x-axis to the eastward direction.

    Returns
    -------
    u_east : Float[jax.Array, "y x"]
        Eastward velocity component
    v_north : Float[jax.Array, "y x"]
        Northward velocity component
    """
    cos_theta = jnp.cos(grid_angle)
    sin_theta = jnp.sin(grid_angle)

    u_east = u * cos_theta - v * sin_theta
    v_north = u * sin_theta + v * cos_theta

    return u_east, v_north


def rotate_to_grid(
    u: Float[jax.Array, "y x"],
    v: Float[jax.Array, "y x"],
    grid_angle: Float[jax.Array, "y x"]
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
    grid_angle : Float[jax.Array, "y x"]
        Angle between the grid x-axis and the eastward direction, in radians. Positive values indicate a counter-clockwise rotation from the grid x-axis to the eastward direction.

    Returns
    -------
    u_grid : Float[jax.Array, "y x"]
        Velocity component along the grid x-axis
    v_grid : Float[jax.Array, "y x"]
        Velocity component along the grid y-axis
    """
    cos_theta = jnp.cos(grid_angle)
    sin_theta = jnp.sin(grid_angle)

    u_grid = u * cos_theta + v * sin_theta
    v_grid = -u * sin_theta + v * cos_theta

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
