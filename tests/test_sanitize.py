import jax.numpy as jnp

from jaxparrow.utils.sanitize import sanitize_data, init_land_mask


class TestSanitizeData:
    def test_replaces_nan_with_fill_value(self):
        arr = jnp.array([[1.0, jnp.nan], [jnp.nan, 4.0]])
        mask = jnp.zeros((2, 2), dtype=bool)
        result = sanitize_data(arr, 0.0, mask)
        expected = jnp.array([[1.0, 0.0], [0.0, 4.0]])
        assert jnp.allclose(result, expected)

    def test_applies_land_mask(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[False, True], [False, False]])
        result = sanitize_data(arr, -999.0, mask)
        assert result[0, 1] == -999.0
        assert result[0, 0] == 1.0
        assert result[1, 0] == 3.0

    def test_replaces_inf_with_fill_value(self):
        arr = jnp.array([[jnp.inf, -jnp.inf], [1.0, 2.0]])
        mask = jnp.zeros((2, 2), dtype=bool)
        result = sanitize_data(arr, 0.0, mask)
        assert jnp.isfinite(result).all()
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0

    def test_nan_fill_value_produces_nan_on_mask(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[False, True], [False, False]])
        result = sanitize_data(arr, jnp.nan, mask)
        assert jnp.isnan(result[0, 1])
        assert result[0, 0] == 1.0

    def test_preserves_valid_data(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.zeros((2, 2), dtype=bool)
        result = sanitize_data(arr, 0.0, mask)
        assert jnp.array_equal(result, arr)


class TestInitLandMask:
    def test_returns_provided_mask(self):
        field = jnp.ones((3, 3))
        mask = jnp.array([[True, False, False],
                          [False, False, False],
                          [False, False, True]])
        result = init_land_mask(field, mask)
        assert jnp.array_equal(result, mask)

    def test_creates_mask_from_nan(self):
        field = jnp.array([[1.0, jnp.nan], [jnp.nan, 4.0]])
        result = init_land_mask(field, None)
        expected = jnp.array([[False, True], [True, False]])
        assert jnp.array_equal(result, expected)

    def test_all_valid_no_mask(self):
        field = jnp.ones((3, 3))
        result = init_land_mask(field, None)
        assert not result.any()

    def test_preserves_shape(self):
        field = jnp.ones((4, 5))
        result = init_land_mask(field, None)
        assert result.shape == (4, 5)

    def test_inf_is_not_finite(self):
        field = jnp.array([[1.0, jnp.inf], [-jnp.inf, 4.0]])
        result = init_land_mask(field, None)
        expected = jnp.array([[False, True], [True, False]])
        assert jnp.array_equal(result, expected)
