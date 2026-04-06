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
        # angle_i = 0 (east), angle_j = pi/2 (north) → output == input
        u = jnp.ones((3, 3))
        v = jnp.zeros((3, 3))
        angle_i = jnp.zeros((3, 3))
        angle_j = jnp.full((3, 3), jnp.pi / 2)
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        assert jnp.allclose(ue, u)
        assert jnp.allclose(vn, v)

    def test_90deg_rotation(self):
        # angle_i = pi/2 (north), angle_j = pi (west): u→north, v→west
        u = jnp.ones((2, 2))
        v = jnp.zeros((2, 2))
        angle_i = jnp.full((2, 2), jnp.pi / 2)
        angle_j = jnp.full((2, 2), jnp.pi)
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        assert jnp.allclose(ue, 0.0, atol=1e-7)
        assert jnp.allclose(vn, u, atol=1e-7)

    def test_left_handed_roundtrip(self):
        # Roundtrip through rotate_to_grid then rotate_to_geographic must recover original
        # geographic velocity for a left-handed grid (angle_i=0, angle_j=-pi/2)
        from jaxparrow.utils.geometry import rotate_to_grid
        u_geo = jnp.ones((2, 2))
        v_geo = jnp.zeros((2, 2))
        angle_i = jnp.zeros((2, 2))
        angle_j = jnp.full((2, 2), -jnp.pi / 2)
        u_grid, v_grid = rotate_to_grid(u_geo, v_geo, angle_i, angle_j)
        ue, vn = rotate_to_geographic(u_grid, v_grid, angle_i, angle_j)
        assert jnp.allclose(ue, u_geo, atol=1e-6)
        assert jnp.allclose(vn, v_geo, atol=1e-6)

    def test_left_handed_u_sign(self):
        # Left-handed grid: angle_i=0 (east), angle_j=-pi/2 (south).
        # ug_t = 1 means _geostrophy computed -g/f * deta_y = 1, where
        # deta_y = ê_j · ∇η = -∂η_n (j-axis points south), so ∂η_n = f/g.
        # Correct u_east = -g/f * ∂η_n = -1.
        # This tests that rotate_to_geographic corrects the sign via the /det factor:
        # without /det → u_east = +1 (wrong); with /det=-1 → u_east = -1 (correct).
        u = jnp.ones((2, 2))   # ug_t = 1
        v = jnp.zeros((2, 2))  # vg_t = 0
        angle_i = jnp.zeros((2, 2))
        angle_j = jnp.full((2, 2), -jnp.pi / 2)
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        assert jnp.allclose(ue, -u, atol=1e-7)   # sign flipped by det=-1
        assert jnp.allclose(vn, 0.0, atol=1e-7)


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
    def test_rectilinear_i_axis_near_zero(self, small_grid):
        lat, lon, _ = small_grid
        angle_i, _ = compute_grid_angle(lat, lon)
        # A regular lat/lon grid has i-axis pointing east → angle_i close to 0
        assert jnp.allclose(angle_i, 0.0, atol=2e-3)

    def test_rectilinear_j_axis_near_pi_over_2(self, small_grid):
        lat, lon, _ = small_grid
        _, angle_j = compute_grid_angle(lat, lon)
        # A regular lat/lon grid (rows increasing northward) has j-axis pointing north → angle_j ≈ π/2
        assert jnp.allclose(angle_j, jnp.pi / 2, atol=2e-3)

    def test_output_shape(self, small_grid):
        lat, lon, _ = small_grid
        angle_i, angle_j = compute_grid_angle(lat, lon)
        assert angle_i.shape == lat.shape
        assert angle_j.shape == lat.shape

    def test_range(self, small_grid):
        lat, lon, _ = small_grid
        angle_i, angle_j = compute_grid_angle(lat, lon)
        assert (angle_i >= -jnp.pi).all()
        assert (angle_i <= jnp.pi).all()
        assert (angle_j >= -jnp.pi).all()
        assert (angle_j <= jnp.pi).all()

    def test_left_handed_grid_det_is_minus_one(self, small_grid):
        lat, lon, _ = small_grid
        # A left-handed grid: flip row order so j-axis points in the opposite direction
        lat_flipped = lat[::-1, :]
        lon_flipped = lon[::-1, :]
        angle_i, angle_j = compute_grid_angle(lat_flipped, lon_flipped)
        # det = sin(angle_j - angle_i) ≈ -1 for a left-handed grid
        det = jnp.cos(angle_i) * jnp.sin(angle_j) - jnp.sin(angle_i) * jnp.cos(angle_j)
        assert jnp.allclose(det, -1.0, atol=2e-3)
        # All angles must be in [-pi, pi]
        assert (angle_j >= -jnp.pi).all()
        assert (angle_j <= jnp.pi).all()


class TestRotateToGrid:
    def test_roundtrip_right_handed(self):
        u = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([[0.5, -0.5], [1.0, -1.0]])
        angle_i = jnp.full((2, 2), jnp.pi / 4)
        angle_j = jnp.full((2, 2), jnp.pi / 4 + jnp.pi / 2)  # right-handed
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        u_rec, v_rec = rotate_to_grid(ue, vn, angle_i, angle_j)
        assert jnp.allclose(u_rec, u, atol=1e-6)
        assert jnp.allclose(v_rec, v, atol=1e-6)

    def test_roundtrip_left_handed(self):
        u = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([[0.5, -0.5], [1.0, -1.0]])
        angle_i = jnp.full((2, 2), jnp.pi / 4)
        angle_j = jnp.full((2, 2), jnp.pi / 4 - jnp.pi / 2)  # left-handed
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        u_rec, v_rec = rotate_to_grid(ue, vn, angle_i, angle_j)
        assert jnp.allclose(u_rec, u, atol=1e-6)
        assert jnp.allclose(v_rec, v, atol=1e-6)

    def test_90deg_rotation(self):
        # angle_i = pi/2 (north), angle_j = pi (west): right-handed
        # u_geo=(1,0): u_grid should give back (1,0) after rotate_to_grid
        u = jnp.ones((2, 2))
        v = jnp.zeros((2, 2))
        angle_i = jnp.full((2, 2), jnp.pi / 2)
        angle_j = jnp.full((2, 2), jnp.pi)
        ug, vg = rotate_to_grid(u, v, angle_i, angle_j)
        ue, vn = rotate_to_geographic(ug, vg, angle_i, angle_j)
        assert jnp.allclose(ue, u, atol=1e-7)
        assert jnp.allclose(vn, v, atol=1e-7)

    def test_inverse_of_rotate_to_geographic(self):
        u = jnp.array([[1.0, 0.0]])
        v = jnp.array([[0.0, 1.0]])
        angle_i = jnp.array([[jnp.pi / 3, jnp.pi / 6]])
        angle_j = angle_i + jnp.pi / 2  # right-handed
        ue, vn = rotate_to_geographic(u, v, angle_i, angle_j)
        u_back, v_back = rotate_to_grid(ue, vn, angle_i, angle_j)
        assert jnp.allclose(u_back, u, atol=1e-6)
        assert jnp.allclose(v_back, v, atol=1e-6)
