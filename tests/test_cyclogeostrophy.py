import jax.numpy as jnp
import optax
import pytest

from jaxparrow import fixed_point, geostrophy, gradient_wind, minimization_based
from jaxparrow.cyclogeostrophy import (
    CyclogeostrophyResult, cyclogeostrophic_loss, cyclogeostrophic_imbalance
)


class TestCyclogeostrophyResult:
    def test_default_none_values(self):
        ucg = jnp.zeros((3, 3))
        vcg = jnp.zeros((3, 3))
        result = CyclogeostrophyResult(ucg=ucg, vcg=vcg)
        assert result.ug is None
        assert result.vg is None
        assert result.ssh is None
        assert result.losses is None

    def test_named_access(self):
        ucg = jnp.ones((3, 3))
        vcg = jnp.ones((3, 3)) * 2
        ug = jnp.ones((3, 3)) * 3
        vg = jnp.ones((3, 3)) * 4
        result = CyclogeostrophyResult(ucg=ucg, vcg=vcg, ug=ug, vg=vg)
        assert jnp.allclose(result.ucg, 1.0)
        assert jnp.allclose(result.vcg, 2.0)
        assert jnp.allclose(result.ug, 3.0)
        assert jnp.allclose(result.vg, 4.0)

    def test_all_fields(self):
        ucg = jnp.zeros((2, 2))
        vcg = jnp.zeros((2, 2))
        ug = jnp.ones((2, 2))
        vg = jnp.ones((2, 2))
        ssh = jnp.ones((2, 2)) * -0.1
        losses = jnp.array([1.0, 0.5, 0.1])
        result = CyclogeostrophyResult(ucg=ucg, vcg=vcg, ug=ug, vg=vg, ssh=ssh, losses=losses)
        assert result.ssh is not None
        assert result.losses is not None
        assert result.losses.shape == (3,)


class TestCyclogeostrophicImbalance:
    def test_output_shapes(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        u_imb, v_imb = cyclogeostrophic_imbalance(u, v, u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert u_imb.shape == lat.shape
        assert v_imb.shape == lat.shape

    def test_uv_on_t_false(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        u_imb, v_imb = cyclogeostrophic_imbalance(
            u, v, u, v, lat_t=lat, lon_t=lon, land_mask=mask, uv_on_t=False
        )
        assert u_imb.shape == lat.shape
        assert v_imb.shape == lat.shape

    def test_infers_grid_from_lat_u(self, small_grid):
        lat, lon, mask = small_grid
        from jaxparrow.utils.geometry import compute_uv_grids
        lat_u, lon_u, _, _ = compute_uv_grids(lat, lon)
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        u_imb, v_imb = cyclogeostrophic_imbalance(
            u, v, u, v, lat_u=lat_u, lon_u=lon_u, land_mask=mask
        )
        assert u_imb.shape == lat.shape

    def test_infers_grid_from_lat_v(self, small_grid):
        lat, lon, mask = small_grid
        from jaxparrow.utils.geometry import compute_uv_grids
        _, _, lat_v, lon_v = compute_uv_grids(lat, lon)
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        u_imb, v_imb = cyclogeostrophic_imbalance(
            u, v, u, v, lat_v=lat_v, lon_v=lon_v, land_mask=mask
        )
        assert u_imb.shape == lat.shape

    def test_raises_without_grids(self):
        u = jnp.ones((3, 3))
        v = jnp.ones((3, 3))
        with pytest.raises(ValueError):
            cyclogeostrophic_imbalance(u, v, u, v)


class TestCyclogeostrophicLoss:
    def test_returns_scalar(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        loss = cyclogeostrophic_loss(u, v, u, v, lat_t=lat, lon_t=lon, land_mask=mask)
        assert loss.shape == ()

    def test_non_negative(self, small_grid):
        lat, lon, mask = small_grid
        u = jnp.ones_like(lat) * 0.1
        v = jnp.ones_like(lat) * 0.1
        loss = cyclogeostrophic_loss(u, v, u * 1.1, v * 1.1, lat_t=lat, lon_t=lon, land_mask=mask)
        assert loss >= 0


class TestFixedPoint:
    def test_from_ssh(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = fixed_point(lat, lon, ssh_t=gaussian_ssh, land_mask=mask)
        assert isinstance(result, CyclogeostrophyResult)
        assert result.ucg.shape == lat.shape
        assert result.vcg.shape == lat.shape

    def test_from_geostrophic(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        result = fixed_point(lat, lon, ug_t=ug, vg_t=vg, land_mask=mask)
        assert result.ucg.shape == lat.shape

    def test_raises_without_inputs(self, small_grid):
        lat, lon, mask = small_grid
        with pytest.raises(ValueError):
            fixed_point(lat, lon, land_mask=mask)

    def test_return_geos(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = fixed_point(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, return_geos=True)
        assert result.ug is not None
        assert result.vg is not None

    def test_return_losses(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        n_it = 5
        result = fixed_point(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, return_losses=True, n_it=n_it)
        assert result.losses is not None
        assert result.losses.shape == (n_it,)

    def test_no_return_geos_by_default(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = fixed_point(lat, lon, ssh_t=gaussian_ssh, land_mask=mask)
        assert result.ug is None
        assert result.vg is None
        assert result.losses is None


class TestGradientWind:
    def test_from_ssh(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = gradient_wind(lat, lon, ssh_t=gaussian_ssh, land_mask=mask)
        assert isinstance(result, CyclogeostrophyResult)
        assert result.ucg.shape == lat.shape

    def test_from_geostrophic(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        result = gradient_wind(lat, lon, ug_t=ug, vg_t=vg, land_mask=mask)
        assert result.ucg.shape == lat.shape

    def test_raises_without_inputs(self, small_grid):
        lat, lon, mask = small_grid
        with pytest.raises(ValueError):
            gradient_wind(lat, lon, land_mask=mask)

    def test_return_geos(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = gradient_wind(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, return_geos=True)
        assert result.ug is not None
        assert result.vg is not None

    def test_no_return_geos_by_default(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = gradient_wind(lat, lon, ssh_t=gaussian_ssh, land_mask=mask)
        assert result.ug is None
        assert result.vg is None


class TestMinimizationBased:
    def test_from_ssh(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = minimization_based(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10)
        assert isinstance(result, CyclogeostrophyResult)
        assert result.ucg.shape == lat.shape

    def test_from_geostrophic(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        result = minimization_based(lat, lon, ug_t=ug, vg_t=vg, land_mask=mask, n_it=10)
        assert result.ucg.shape == lat.shape

    def test_raises_without_inputs(self, small_grid):
        lat, lon, mask = small_grid
        with pytest.raises(ValueError):
            minimization_based(lat, lon, land_mask=mask, n_it=10)

    def test_return_geos(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = minimization_based(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, return_geos=True, n_it=10)
        assert result.ug is not None
        assert result.vg is not None

    def test_return_losses(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        n_it = 10
        result = minimization_based(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, return_losses=True, n_it=n_it)
        assert result.losses is not None
        assert result.losses.shape == (n_it,)

    def test_custom_optimizer_string(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = minimization_based(
            lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10,
            optim="adam", optim_kwargs={"learning_rate": 0.01}
        )
        assert result.ucg.shape == lat.shape

    def test_custom_optax_optimizer(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = minimization_based(
            lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10,
            optim=optax.adam(learning_rate=0.01)
        )
        assert result.ucg.shape == lat.shape

    def test_invalid_optimizer_raises(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        with pytest.raises(TypeError):
            minimization_based(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10, optim=42)

    def test_no_return_geos_by_default(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid
        result = minimization_based(lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10)
        assert result.ug is None
        assert result.vg is None
        assert result.losses is None

    def test_regularization_system_params_only(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ucg_t, vcg_t):
            return jnp.sum(ucg_t ** 2 + vcg_t ** 2) * 0.01

        result = minimization_based(
            lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10, regularization=reg
        )
        assert result.ucg.shape == lat.shape

    def test_regularization_with_user_kwargs(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ucg_t, alpha):
            return jnp.sum(ucg_t ** 2) * alpha

        result = minimization_based(
            lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10,
            regularization=reg, reg_kwargs={"alpha": jnp.array(0.01)}
        )
        assert result.ucg.shape == lat.shape

    def test_regularization_ssh_optimization(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ssh_t):
            return jnp.sum(ssh_t ** 2) * 0.01

        result = minimization_based(
            lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10, regularization=reg
        )
        assert result.ucg.shape == lat.shape
        assert result.ssh is not None
        assert result.ssh.shape == lat.shape

    def test_regularization_geos_optimization(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ug_t, vg_t):
            return jnp.sum(ug_t ** 2 + vg_t ** 2) * 0.01

        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        result = minimization_based(
            lat, lon, ug_t=ug, vg_t=vg, land_mask=mask, n_it=10, regularization=reg
        )
        assert result.ucg.shape == lat.shape
        assert result.ug is not None
        assert result.vg is not None

    def test_regularization_ssh_without_ssh_raises(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ssh_t):
            return jnp.sum(ssh_t ** 2) * 0.01

        ug, vg = geostrophy(gaussian_ssh, lat, lon, land_mask=mask)
        with pytest.raises(ValueError, match="ssh_t"):
            minimization_based(
                lat, lon, ug_t=ug, vg_t=vg, land_mask=mask, n_it=10, regularization=reg
            )

    def test_regularization_missing_reg_kwargs_raises(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ucg_t, alpha):
            return jnp.sum(ucg_t ** 2) * alpha

        with pytest.raises(ValueError, match="reg_kwargs"):
            minimization_based(
                lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10, regularization=reg
            )

    def test_regularization_incomplete_reg_kwargs_raises(self, small_grid, gaussian_ssh):
        lat, lon, mask = small_grid

        def reg(ucg_t, alpha, beta):
            return jnp.sum(ucg_t ** 2) * alpha * beta

        with pytest.raises(ValueError, match="not found in reg_kwargs"):
            minimization_based(
                lat, lon, ssh_t=gaussian_ssh, land_mask=mask, n_it=10,
                regularization=reg, reg_kwargs={"alpha": jnp.array(0.01)}
            )


class TestAnalyticalGaussianEddy:
    """Tests against analytical Gaussian eddy solutions."""

    def test_cyclogeostrophic_imbalance_near_zero(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        u_imb, v_imb = cyclogeostrophic_imbalance(
            d["ug"], d["vg"], d["ucg"], d["vcg"],
            lat_t=d["lat"], lon_t=d["lon"], land_mask=d["land_mask"]
        )

        # Analytical velocities should nearly satisfy the cyclogeostrophic balance
        V = jnp.sqrt(d["ucg"] ** 2 + d["vcg"] ** 2)
        imb_magnitude = jnp.sqrt(u_imb ** 2 + v_imb ** 2)
        rel_imb = imb_magnitude[d["annular"]] / V[d["annular"]]
        assert jnp.nanmedian(rel_imb) < 0.05

    def test_cyclogeostrophic_loss_near_zero(self, gaussian_eddy_data):
        d = gaussian_eddy_data
        loss = cyclogeostrophic_loss(
            d["ug"], d["vg"], d["ucg"], d["vcg"],
            lat_t=d["lat"], lon_t=d["lon"], land_mask=d["land_mask"]
        )

        # Loss from analytical solution should be small relative to a
        # deliberately imbalanced field (using geostrophic as cyclogeostrophic)
        loss_imbalanced = cyclogeostrophic_loss(
            d["ug"], d["vg"], d["ug"], d["vg"],
            lat_t=d["lat"], lon_t=d["lon"], land_mask=d["land_mask"]
        )
        assert loss < loss_imbalanced
