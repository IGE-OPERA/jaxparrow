import jax
import jax.numpy as jnp

from jaxparrow import fixed_point, geostrophy, gradient_wind, minimization_based

import gaussian_eddy


jax.config.update("jax_enable_x64", True)


class TestVelocities:
    lat, lon, ssh, ug, vg, ucg, vcg, land_mask = gaussian_eddy.simulate_gaussian_eddy(R0=50e3, eta0=-.2)

    def test_geostrophy(self):
        ug_est, vg_est = geostrophy(
            ssh_t=self.ssh, lat_t=self.lat, lon_t=self.lon, land_mask=self.land_mask, 
            rotate_to_geographic=False
        )
        geos_diff = self.compute_diff(self.ug, self.vg, ug_est, vg_est)  # around 0.00045
        assert geos_diff < .0005

    def test_cyclogeostrophy_fixed_point(self):
        res_fp = fixed_point(
            lat_t=self.lat, lon_t=self.lon, ug_t=self.ug, vg_t=self.vg, land_mask=self.land_mask, 
            rotate_to_geographic=False
        )

        ucg_est, vcg_est = res_fp.ucg, res_fp.vcg
        fp_diff = self.compute_diff(self.ucg, self.vcg, ucg_est, vcg_est)  # around 0.00034
        assert fp_diff < .0004

    def test_cyclogeostrophy_gradient_wind(self):
        res_gw = gradient_wind(
            lat_t=self.lat, lon_t=self.lon, ug_t=self.ug, vg_t=self.vg, land_mask=self.land_mask, 
            rotate_to_geographic=False
        )

        ucg_est, vcg_est = res_gw.ucg, res_gw.vcg
        gw_diff = self.compute_diff(self.ucg, self.vcg, ucg_est, vcg_est)  # around 1.25e-05
        assert gw_diff < 1.3e-5

    def test_cyclogeostrophy_minimization(self):
        res_mb = minimization_based(
            lat_t=self.lat, lon_t=self.lon, ug_t=self.ug, vg_t=self.vg, land_mask=self.land_mask, 
            rotate_to_geographic=False
        )

        ucg_est, vcg_est = res_mb.ucg, res_mb.vcg
        mb_diff = self.compute_diff(self.ucg, self.vcg, ucg_est, vcg_est)  # around 6.17e-05
        assert mb_diff < 6.2e-5

    @staticmethod
    def compute_diff(u, v, u_est, v_est) -> float:
        x = jnp.stack([u, v], axis=-1)
        x_est = jnp.stack([u_est, v_est], axis=-1)
        return jnp.sqrt(jnp.nanmean((x - x_est)**2))
