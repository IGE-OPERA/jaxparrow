import jax.numpy as jnp
import pytest

from jaxparrow.utils.geometry import GRAVITY
from jaxparrow.utils.kinematics import (
    magnitude, kinetic_energy, vorticity, strain_rate, radius_of_curvature, setup_kinematics
)


class TestMagnitude:
    def test_simple_values(self):
        u = jnp.array([[3.0, 0.0], [0.0, 1.0]])
        v = jnp.array([[4.0, 1.0], [0.0, 0.0]])
        result = magnitude(u, v)
        expected = jnp.array([[5.0, 1.0], [0.0, 1.0]])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_zero_velocity(self):
        u = jnp.zeros((3, 3))
        v = jnp.zeros((3, 3))
        result = magnitude(u, v)
        assert jnp.allclose(result, 0.0)

    def test_land_mask_produces_nan(self):
        u = jnp.ones((3, 3))
        v = jnp.ones((3, 3))
        mask = jnp.array([[False, True, False],
                          [False, False, False],
                          [False, False, False]])
        result = magnitude(u, v, land_mask=mask)
        assert jnp.isnan(result[0, 1])
        assert not jnp.isnan(result[0, 0])

    def test_preserves_shape(self):
        u = jnp.ones((4, 5))
        v = jnp.ones((4, 5))
        assert magnitude(u, v).shape == (4, 5)

    def test_uv_on_t_false(self):
        u = jnp.ones((4, 5)) * 3.0
        v = jnp.ones((4, 5)) * 4.0
        result = magnitude(u, v, uv_on_t=False)
        assert result.shape == (4, 5)
        # after interpolation from U/V to T, uniform fields stay uniform
        assert jnp.allclose(result, 5.0, atol=1e-5)


class TestKineticEnergy:
    def test_simple_values(self):
        u = jnp.array([[3.0, 0.0]])
        v = jnp.array([[4.0, 1.0]])
        result = kinetic_energy(u, v)
        expected = jnp.array([[12.5, 0.5]])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_zero_velocity(self):
        u = jnp.zeros((3, 3))
        v = jnp.zeros((3, 3))
        result = kinetic_energy(u, v)
        assert jnp.allclose(result, 0.0)

    def test_land_mask_produces_nan(self):
        u = jnp.ones((3, 3))
        v = jnp.ones((3, 3))
        mask = jnp.array([[True, False, False],
                          [False, False, False],
                          [False, False, False]])
        result = kinetic_energy(u, v, land_mask=mask)
        assert jnp.isnan(result[0, 0])

    def test_preserves_shape(self):
        u = jnp.ones((4, 5))
        v = jnp.ones((4, 5))
        assert kinetic_energy(u, v).shape == (4, 5)

    def test_uv_on_t_false(self):
        u = jnp.ones((4, 5)) * 3.0
        v = jnp.ones((4, 5)) * 4.0
        result = kinetic_energy(u, v, uv_on_t=False)
        assert result.shape == (4, 5)
        assert jnp.allclose(result, 12.5, atol=1e-5)


class TestVorticity:
    def test_output_shape(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.zeros_like(lat)
        result = vorticity(u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert result.shape == lat.shape

    def test_uniform_flow_zero_vorticity(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.5
        v = jnp.ones_like(lat) * 0.5
        result = vorticity(u, v, lat_t=lat, lon_t=lon, land_mask=mask, normalize_by_coriolis=False)
        inner = result[2:-2, 2:-2]
        assert jnp.allclose(inner, 0.0, atol=1e-5)

    def test_coriolis_normalization_flag(self, small_grid):
        lat, lon, mask = small_grid
        # create a sheared flow that produces non-zero vorticity
        u = jnp.zeros_like(lat)
        v = jnp.broadcast_to(jnp.linspace(-0.1, 0.1, lat.shape[1]), lat.shape)
        result_norm = vorticity(u, v, lat_t=lat, lon_t=lon, land_mask=mask, normalize_by_coriolis=True)
        result_raw = vorticity(u, v, lat_t=lat, lon_t=lon, land_mask=mask, normalize_by_coriolis=False)
        # normalized values should differ from raw values
        assert not jnp.allclose(result_norm[3:-3, 3:-3], result_raw[3:-3, 3:-3])


class TestStrainRate:
    def test_output_shape(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.zeros_like(lat)
        result = strain_rate(u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert result.shape == lat.shape

    def test_uniform_flow_zero_strain(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.5
        v = jnp.ones_like(lat) * 0.5
        result = strain_rate(u, v, lat_t=lat, lon_t=lon, land_mask=mask, normalize_by_coriolis=False)
        inner = result[2:-2, 2:-2]
        assert jnp.allclose(inner, 0.0, atol=1e-5)


class TestRadiusOfCurvature:
    def test_output_shape(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.broadcast_to(jnp.linspace(-0.1, 0.1, lat.shape[1]), lat.shape)
        result = radius_of_curvature(u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert result.shape == lat.shape


class TestSetupKinematics:
    def test_returns_correct_number_of_outputs(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        result = setup_kinematics(u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert len(result) == 10

    def test_raises_without_grids(self):
        u = jnp.ones((3, 3))
        v = jnp.ones((3, 3))
        with pytest.raises(ValueError):
            setup_kinematics(u, v)

    def test_infers_lat_lon_from_u_grid(self, small_grid):
        lat, lon, mask = small_grid
        from jaxparrow.utils.geometry import compute_uv_grids
        lat_u, lon_u, _, _ = compute_uv_grids(lat, lon)
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        result = setup_kinematics(u, v, lat_u=lat_u, lon_u=lon_u, land_mask=mask)
        assert len(result) == 10
        # lat_t and lon_t should have been inferred
        assert result[2].shape == lat.shape  # lat_t
        assert result[3].shape == lon.shape  # lon_t

    def test_uv_on_t_false_interpolates(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        result = setup_kinematics(u, v, lat_t=lat, lon_t=lon, land_mask=mask, uv_on_t=False)
        assert len(result) == 10
        # u and v should have been interpolated but still same shape
        assert result[0].shape == lat.shape
        assert result[1].shape == lat.shape

    def test_infers_lat_lon_from_v_grid(self, small_grid):
        lat, lon, mask = small_grid
        from jaxparrow.utils.geometry import compute_uv_grids
        _, _, lat_v, lon_v = compute_uv_grids(lat, lon)
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        result = setup_kinematics(u, v, lat_v=lat_v, lon_v=lon_v, land_mask=mask)
        assert len(result) == 10
        assert result[2].shape == lat.shape
        assert result[3].shape == lon.shape


class TestAnalyticalGaussianEddy:
    """Tests against analytical solutions for a Gaussian eddy."""

    def test_vorticity_matches_analytical(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        vort = vorticity(
            d["ug"], d["vg"], lat_t=d["lat"], lon_t=d["lon"],
            land_mask=d["land_mask"], normalize_by_coriolis=False
        )

        # Analytical: ζ = 2A(r²/R₀² - 1) where A = 2gη/(fR₀²)
        A = 2 * GRAVITY * d["ssh"] / (d["f"] * d["R0"] ** 2)
        vort_analytical = 2 * A * (d["R"] ** 2 / d["R0"] ** 2 - 1)

        rel_error = jnp.abs(vort[d["interior"]] - vort_analytical[d["interior"]]) / jnp.abs(vort_analytical[d["interior"]])
        assert jnp.nanmedian(rel_error) < 0.05

    def test_strain_rate_matches_analytical(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        sr = strain_rate(
            d["ug"], d["vg"], lat_t=d["lat"], lon_t=d["lon"],
            land_mask=d["land_mask"], normalize_by_coriolis=False
        )

        # Analytical: strain = 2|A|r²/R₀² where A = 2gη/(fR₀²)
        A = 2 * GRAVITY * d["ssh"] / (d["f"] * d["R0"] ** 2)
        strain_analytical = 2 * jnp.abs(A) * d["R"] ** 2 / d["R0"] ** 2

        rel_error = jnp.abs(sr[d["annular"]] - strain_analytical[d["annular"]]) / strain_analytical[d["annular"]]
        assert jnp.nanmedian(rel_error) < 0.05

    def test_radius_of_curvature_matches_analytical(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        rc = radius_of_curvature(
            d["ug"], d["vg"], lat_t=d["lat"], lon_t=d["lon"],
            land_mask=d["land_mask"]
        )

        # Analytical: R_c = r for a purely azimuthal Gaussian eddy flow
        rel_error = jnp.abs(rc[d["annular"]] - d["R"][d["annular"]]) / d["R"][d["annular"]]
        assert jnp.nanmedian(rel_error) < 0.05
