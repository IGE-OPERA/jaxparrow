import jax.numpy as jnp

from jaxparrow.utils.geometry import (
    coriolis_factor, grid_metrics, compute_uv_grids, EARTH_ANG_SPEED
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


class TestGridMetrics:
    def test_output_shapes(self, small_grid):
        lat, lon, _ = small_grid
        dx_e, dx_n, dy_e, dy_n, J = grid_metrics(lat, lon)
        assert dx_e.shape == lat.shape
        assert dx_n.shape == lat.shape
        assert dy_e.shape == lat.shape
        assert dy_n.shape == lat.shape
        assert J.shape == lat.shape

    def test_regular_grid_dominant_components(self, small_grid):
        lat, lon, _ = small_grid
        dx_e, dx_n, dy_e, dy_n, _ = grid_metrics(lat, lon)
        # For a regular lat/lon grid:
        # stepping in x (lon) -> mostly eastward: |dx_e| >> |dx_n|
        # stepping in y (lat) -> mostly northward: |dy_n| >> |dy_e|
        inner = slice(1, -1), slice(1, -1)
        assert jnp.abs(dx_e[inner]).mean() > jnp.abs(dx_n[inner]).mean() * 10
        assert jnp.abs(dy_n[inner]).mean() > jnp.abs(dy_e[inner]).mean() * 10

    def test_positive_jacobian(self, small_grid):
        lat, lon, _ = small_grid
        _, _, _, _, J = grid_metrics(lat, lon)
        assert (J > 0).all()

    def test_displacements_in_meters(self, small_grid):
        lat, lon, _ = small_grid
        dx_e, _, _, dy_n, _ = grid_metrics(lat, lon)
        # 1° of latitude ~ 111 km, grid spacing ~ 0.11° -> ~ 12 km
        # 1° of longitude at 36°N ~ 90 km, grid spacing ~ 0.11° -> ~ 10 km
        inner = slice(1, -1), slice(1, -1)
        assert jnp.abs(dx_e[inner]).mean() > 5000   # > 5 km
        assert jnp.abs(dx_e[inner]).mean() < 20000  # < 20 km
        assert jnp.abs(dy_n[inner]).mean() > 5000
        assert jnp.abs(dy_n[inner]).mean() < 20000


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
