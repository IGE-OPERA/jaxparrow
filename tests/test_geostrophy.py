import jax
import jax.numpy as jnp

from jaxparrow import geostrophy
from jaxparrow.utils.geometry import compute_grid_angle, rotate_to_geographic


jax.config.update("jax_enable_x64", True)


class TestGeostrophy:
    def test_output_shapes(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        assert ug.shape == lat.shape
        assert vg.shape == lat.shape

    def test_flat_ssh_zero_velocity(self, small_grid):
        lat, lon, mask = small_grid
        ssh = jnp.zeros_like(lat)
        ug, vg = geostrophy(ssh, lat, lon, land_mask=mask)
        assert jnp.allclose(ug, 0.0, atol=1e-10)
        assert jnp.allclose(vg, 0.0, atol=1e-10)

    def test_land_mask_produces_nan(self, small_grid, gaussian_ssh):
        lat, lon, _ = small_grid
        mask = jnp.zeros_like(lat, dtype=bool).at[3, 3].set(True)
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        assert jnp.isnan(ug[3, 3])
        assert jnp.isnan(vg[3, 3])

    def test_infers_mask_from_nan(self, small_grid):
        lat, lon, _ = small_grid
        ssh = jnp.ones_like(lat)
        ssh = ssh.at[0, 0].set(jnp.nan)
        ug, vg = geostrophy(ssh, lat, lon)
        assert jnp.isnan(ug[0, 0])
        assert jnp.isnan(vg[0, 0])

    def test_rotate_to_geographic_false(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        # Force rectilinear so rotation is identity; True vs False should give identical outputs
        ug_geo, vg_geo = geostrophy(gaussian_ssh, lat, lon, land_mask=mask,
                                     is_grid_rectilinear=True, rotate_to_geographic=True)
        ug_grid, vg_grid = geostrophy(gaussian_ssh, lat, lon, land_mask=mask,
                                       is_grid_rectilinear=True, rotate_to_geographic=False)
        assert jnp.allclose(ug_geo, ug_grid, equal_nan=True)
        assert jnp.allclose(vg_geo, vg_grid, equal_nan=True)

    def test_is_grid_rectilinear_true(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask, is_grid_rectilinear=True)
        assert ug.shape == lat.shape
        assert vg.shape == lat.shape
        assert not jnp.all(jnp.isnan(ug))

    def test_is_grid_rectilinear_false_computes_rotation(self, small_grid, gaussian_ssh):
        # is_grid_rectilinear=False with rotate_to_geographic=True (default) forces the code
        # to compute the grid angle on demand (covers the branch where grid_angle is None)
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask, is_grid_rectilinear=False)
        assert ug.shape == lat.shape
        assert vg.shape == lat.shape

    def test_non_trivial_ssh_produces_nonzero_velocity(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        inner = slice(2, -2), slice(2, -2)
        assert not jnp.allclose(ug[inner], 0.0)
        assert not jnp.allclose(vg[inner], 0.0)

    def test_rotate_to_geographic_true_vs_false_curvilinear(self, small_curvilinear_grid):
        lat, lon, mask = small_curvilinear_grid
        # Recompute gaussian_ssh for this grid
        R = jnp.sqrt((lat - 36.0) ** 2 + (lon + 4.5) ** 2)
        ssh = -0.2 * jnp.exp(-(R / 0.3) ** 2)
        ug_geo, vg_geo = geostrophy(ssh, lat, lon, land_mask=mask, rotate_to_geographic=True)
        ug_grid, vg_grid = geostrophy(ssh, lat, lon, land_mask=mask, rotate_to_geographic=False)
        # Should differ for curvilinear grid
        assert not jnp.allclose(ug_geo, ug_grid) or not jnp.allclose(vg_geo, vg_grid)
        # Rotating grid-relative back to geo should recover geo
        angle_i, angle_j = compute_grid_angle(lat, lon)
        ug_geo2, vg_geo2 = rotate_to_geographic(ug_grid, vg_grid, angle_i, angle_j)
        assert jnp.allclose(ug_geo, ug_geo2, atol=1e-6)
        assert jnp.allclose(vg_geo, vg_geo2, atol=1e-6)


class TestAnalyticalGaussianEddy:
    """Tests numerical geostrophy against the analytical Gaussian eddy solution."""

    def test_geostrophy_matches_analytical(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        ug_num, vg_num = geostrophy(d["ssh"], d["lat"], d["lon"], land_mask=d["land_mask"])

        # RMSE between numerical and analytical geostrophic velocities
        diff = jnp.sqrt(jnp.nanmean((ug_num - d["ug"]) ** 2 + (vg_num - d["vg"]) ** 2))
        assert diff < 0.001

    def test_geostrophy_relative_error_in_interior(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        ug_num, vg_num = geostrophy(d["ssh"], d["lat"], d["lon"], land_mask=d["land_mask"])

        V_analytical = jnp.sqrt(d["ug"] ** 2 + d["vg"] ** 2)
        V_numerical = jnp.sqrt(ug_num ** 2 + vg_num ** 2)
        rel_error = jnp.abs(V_numerical - V_analytical) / V_analytical

        assert jnp.nanmedian(rel_error[d["annular"]]) < 0.02
