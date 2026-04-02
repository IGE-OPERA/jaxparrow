import jax.numpy as jnp
import pytest

from jaxparrow.utils.operators import interpolation, derivative, horizontal_derivatives


class TestInterpolation:
    def test_uniform_field_unchanged(self):
        field = jnp.ones((3, 4))
        result = interpolation(field, axis=1, padding="right")
        assert jnp.allclose(result, 1.0)
        assert result.shape == field.shape

    def test_axis1_right_padding(self):
        field = jnp.array([[1.0, 3.0, 5.0, 7.0]])
        result = interpolation(field, axis=1, padding="right")
        # midpoints: [2, 4, 6], padded right: [2, 4, 6, 6]
        expected = jnp.array([[2.0, 4.0, 6.0, 6.0]])
        assert jnp.allclose(result, expected)

    def test_axis1_left_padding(self):
        field = jnp.array([[1.0, 3.0, 5.0, 7.0]])
        result = interpolation(field, axis=1, padding="left")
        # midpoints: [2, 4, 6], padded left: [2, 2, 4, 6]
        expected = jnp.array([[2.0, 2.0, 4.0, 6.0]])
        assert jnp.allclose(result, expected)

    def test_axis0_right_padding(self):
        field = jnp.array([[1.0], [3.0], [5.0]])
        result = interpolation(field, axis=0, padding="right")
        expected = jnp.array([[2.0], [4.0], [4.0]])
        assert jnp.allclose(result, expected)

    def test_axis0_left_padding(self):
        field = jnp.array([[1.0], [3.0], [5.0]])
        result = interpolation(field, axis=0, padding="left")
        expected = jnp.array([[2.0], [2.0], [4.0]])
        assert jnp.allclose(result, expected)

    def test_nan_one_valid_uses_valid(self):
        field = jnp.array([[1.0, jnp.nan, 5.0]])
        result = interpolation(field, axis=1, padding="right")
        # (1.0, nan) -> use left=1.0; (nan, 5.0) -> use right=5.0
        assert result[0, 0] == 1.0
        assert result[0, 1] == 5.0

    def test_nan_both_nan_stays_nan(self):
        field = jnp.array([[jnp.nan, jnp.nan, 5.0]])
        result = interpolation(field, axis=1, padding="right")
        assert jnp.isnan(result[0, 0])

    def test_land_mask_sets_nan(self):
        field = jnp.array([[1.0, 3.0, 5.0]])
        mask = jnp.array([[False, True, False]])
        result = interpolation(field, axis=1, padding="right", land_mask=mask)
        assert jnp.isnan(result[0, 1])

    def test_preserves_shape(self):
        field = jnp.ones((5, 7))
        for axis in [0, 1]:
            for padding in ["left", "right"]:
                result = interpolation(field, axis=axis, padding=padding)
                assert result.shape == field.shape


class TestDerivative:
    def test_constant_field_zero_derivative(self):
        field = jnp.full((3, 4), 5.0)
        result = derivative(field, axis=1,)
        assert jnp.allclose(result, 0.0)

    def test_linear_field_constant_derivative(self):
        field = jnp.array([[1.0, 3.0, 5.0, 7.0]])
        result = derivative(field, axis=1)
        assert jnp.allclose(result, 2.0)

    def test_axis0_derivative(self):
        field = jnp.array([[1.0], [4.0], [9.0]])
        result = derivative(field, axis=0)
        # diff: [3, 5], padding right: [3, 5, 5]
        expected = jnp.array([[3.0], [5.0], [5.0]])
        assert jnp.allclose(result, expected)

    def test_preserves_shape(self):
        field = jnp.ones((5, 7))
        for axis in [0, 1]:
            result = derivative(field, axis=axis)
            assert result.shape == field.shape

    def test_land_mask_sets_nan(self):
        field = jnp.array([[1.0, 3.0, 5.0]])
        mask = jnp.array([[False, True, False]])
        result = derivative(field, axis=1, land_mask=mask)
        assert jnp.isnan(result[0, 1])


class TestHorizontalDerivatives:
    def test_from_lat_lon(self, small_grid):
        lat, lon, _ = small_grid
        field = lat
        df_e, df_n = horizontal_derivatives(field, lat=lat, lon=lon)
        assert df_e.shape == field.shape
        assert df_n.shape == field.shape

    def test_from_grid_spacing(self, small_grid):
        lat, lon, _ = small_grid
        from jaxparrow.utils.geometry import grid_spacing
        dx, dy = grid_spacing(lat, lon)
        field = lat
        # Assume horizontal_derivatives can take dx, dy for orthogonal grid
        df_e, df_n = horizontal_derivatives(field, dx=dx, dy=dy)
        assert df_e.shape == field.shape
        assert df_n.shape == field.shape

    def test_raises_without_inputs(self):
        field = jnp.ones((3, 3))
        with pytest.raises(ValueError):
            horizontal_derivatives(field)

    def test_constant_field_zero_gradient(self, small_grid):
        lat, lon, _ = small_grid
        field = jnp.ones_like(lat) * 5.0
        df_e, df_n = horizontal_derivatives(field, lat=lat, lon=lon)
        assert jnp.allclose(df_e, 0.0, atol=1e-10)
        assert jnp.allclose(df_n, 0.0, atol=1e-10)
