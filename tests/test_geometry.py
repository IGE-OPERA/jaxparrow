import jax.numpy as jnp

from jaxparrow.utils.geometry import (
    coriolis_factor, compute_uv_grids, compute_grid_angle,
    EARTH_ANG_SPEED, grid_spacing, rotate_to_geographic, rotate_to_grid
)


class TestCoriolisFactory:
    def test_equator_zero(self):
        lat = jnp.array([[0.0]])
        f = coriolis_factor(lat)
        assert jnp.allclose(f, 0.0, atol=1e-10)

    def test_north_pole_max(self):
        lat = jnp.array([[90.0]])
        f = coriolis_factor(lat)
        expected = 2 * EARTH_ANG_SPEED
        assert jnp.allclose(f, expected, rtol=1e-5)

    def test_positive_in_northern_hemisphere(self):
        lat = jnp.array([[45.0]])
        f = coriolis_factor(lat)
        assert f > 0

    def test_negative_in_southern_hemisphere(self):
        lat = jnp.array([[-45.0]])
        f = coriolis_factor(lat)
        assert f < 0

    def test_antisymmetric(self):
        lat_n = jnp.array([[30.0]])
        lat_s = jnp.array([[-30.0]])
        assert jnp.allclose(coriolis_factor(lat_n), -coriolis_factor(lat_s))

    def test_preserves_shape(self):
        lat = jnp.ones((3, 5)) * 45.0
        f = coriolis_factor(lat)
        assert f.shape == (3, 5)


class TestGridSpacing:
    def test_output_shapes(self, small_grid):
        lat, lon, _ = small_grid
        dx, dy = grid_spacing(lat, lon)
        assert dx.shape == lat.shape
        assert dy.shape == lat.shape

    def test_physical_range(self, small_grid):
        lat, lon, _ = small_grid
        dx, dy = grid_spacing(lat, lon)
        inner = slice(1, -1), slice(1, -1)
        # 1 degree ~ 111 km, grid ~0.11 deg -> ~12 km
        assert jnp.abs(dx[inner]).mean() > 5000
        assert jnp.abs(dx[inner]).mean() < 20000
        assert jnp.abs(dy[inner]).mean() > 5000
        assert jnp.abs(dy[inner]).mean() < 20000


class TestRotateToGeographic:
    def test_identity_rotation(self):
        # grid_angle = 0, so output == input
        u = jnp.ones((3, 3))
        v = jnp.zeros((3, 3))
        angle = jnp.zeros((3, 3))
        ue, vn = rotate_to_geographic(u, v, angle)
        assert jnp.allclose(ue, u)
        assert jnp.allclose(vn, v)

    def test_90deg_rotation(self):
        # grid_angle = pi/2, so u becomes -v, v becomes u
        u = jnp.ones((2, 2))
        v = jnp.zeros((2, 2))
        angle = jnp.ones((2, 2)) * (jnp.pi / 2)
        ue, vn = rotate_to_geographic(u, v, angle)
        assert jnp.allclose(ue, 0.0, atol=1e-7)
        assert jnp.allclose(vn, u, atol=1e-7)


class TestComputeUVGrids:
    def test_output_shapes(self, small_grid):
        lat, lon, _ = small_grid
        lat_u, lon_u, lat_v, lon_v = compute_uv_grids(lat, lon)
        assert lat_u.shape == lat.shape
        assert lon_u.shape == lon.shape
        assert lat_v.shape == lat.shape
        assert lon_v.shape == lon.shape

    def test_u_grid_shifted_in_x(self, small_grid):
        lat, lon, _ = small_grid
        _, lon_u, _, _ = compute_uv_grids(lat, lon)
        # U grid lon should be midpoint of consecutive T grid lon values
        mid_lon = (lon[:, :-1] + lon[:, 1:]) / 2
        assert jnp.allclose(lon_u[:, :-1], mid_lon, atol=1e-5)

    def test_v_grid_shifted_in_y(self, small_grid):
        lat, lon, _ = small_grid
        _, _, lat_v, _ = compute_uv_grids(lat, lon)
        # V grid lat should be midpoint of consecutive T grid lat values
        mid_lat = (lat[:-1, :] + lat[1:, :]) / 2
        assert jnp.allclose(lat_v[:-1, :], mid_lat, atol=1e-5)

    def test_boundary_extrapolation(self, small_grid):
        lat, lon, _ = small_grid
        lat_u, lon_u, lat_v, lon_v = compute_uv_grids(lat, lon)
        # Last column of U grid extrapolated: 2*T[-1] - T[-2]
        expected_lon_u_last = 2 * lon[:, -1] - lon[:, -2]
        assert jnp.allclose(lon_u[:, -1], expected_lon_u_last)
        # Last row of V grid extrapolated: 2*T[-1] - T[-2]
        expected_lat_v_last = 2 * lat[-1, :] - lat[-2, :]
        assert jnp.allclose(lat_v[-1, :], expected_lat_v_last)


class TestComputeGridAngle:
    def test_rectilinear_grid_near_zero(self, small_grid):
        lat, lon, _ = small_grid
        angle = compute_grid_angle(lat, lon)
        # A regular lat/lon grid has i-axis pointing east → angle close to 0
        assert jnp.allclose(angle, 0.0, atol=2e-3)

    def test_output_shape(self, small_grid):
        lat, lon, _ = small_grid
        angle = compute_grid_angle(lat, lon)
        assert angle.shape == lat.shape

    def test_range(self, small_grid):
        lat, lon, _ = small_grid
        angle = compute_grid_angle(lat, lon)
        assert (angle >= -jnp.pi).all()
        assert (angle <= jnp.pi).all()


class TestRotateToGrid:
    def test_roundtrip(self):
        u = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([[0.5, -0.5], [1.0, -1.0]])
        angle = jnp.ones((2, 2)) * (jnp.pi / 4)
        ue, vn = rotate_to_geographic(u, v, angle)
        u_rec, v_rec = rotate_to_grid(ue, vn, angle)
        assert jnp.allclose(u_rec, u, atol=1e-6)
        assert jnp.allclose(v_rec, v, atol=1e-6)

    def test_90deg_rotation(self):
        u = jnp.ones((2, 2))
        v = jnp.zeros((2, 2))
        angle = jnp.ones((2, 2)) * (jnp.pi / 2)
        # rotate_to_grid: u_grid = u*cos + v*sin, v_grid = -u*sin + v*cos
        # with cos=0, sin=1: u_grid = 0, v_grid = -u = -1
        ug, vg = rotate_to_grid(u, v, angle)
        assert jnp.allclose(ug, 0.0, atol=1e-7)
        assert jnp.allclose(vg, -u, atol=1e-7)

    def test_inverse_of_rotate_to_geographic(self):
        u = jnp.array([[1.0, 0.0]])
        v = jnp.array([[0.0, 1.0]])
        angle = jnp.array([[jnp.pi / 3, jnp.pi / 6]])
        ue, vn = rotate_to_geographic(u, v, angle)
        u_back, v_back = rotate_to_grid(ue, vn, angle)
        assert jnp.allclose(u_back, u, atol=1e-6)
        assert jnp.allclose(v_back, v, atol=1e-6)
