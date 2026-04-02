from typing import Literal

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float


def interpolation(
    field: Float[jax.Array, "y x"],
    axis: Literal[0, 1],
    padding: Literal["left", "right"],
    land_mask: Float[jax.Array, "y x"] = None,
) -> Float[jax.Array, "y x"]:
    """
    Interpolates the values of a ``field`` along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/`x`),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain.

    An open boundary condition is applied:

    - At domain edges: the interpolated value equals the nearest interior value
    - At land/NaN boundaries: if one of the two values is NaN, the valid value is used;
      if both are NaN, the result is NaN

    Parameters
    ----------
    field : Float[jax.Array, "y x"]
        Field to interpolate
    axis : Literal[0, 1]
        Axis along which interpolation is performed
    padding : Literal["left", "right"]
        Padding direction.
        For example, following NEMO convention,
        interpolating from U to T points requires a `left` padding
        (the midpoint between $U_i$ and $U_{i+1}$ corresponds to $T_{i+1}$),
        and interpolating from T to U points a `right` padding
        (the midpoint between $T_i$ and $T_{i+1}$ corresponds to $U_i$)
    land_mask : Float[jax.Array, "y x"], optional
        Mask indicating the land domain where extrapolation should be applied.
        `False`/`0` indicates ocean cells, `True`/`1` indicates land cells.

        Defaults to `None`, 
        in which case no land masking is applied and extrapolation is performed across the entire domain

    Returns
    -------
    field : Float[jax.Array, "y x"]
        Interpolated field
    """
    f = jnp.moveaxis(field, axis, -1)

    left = f[:, :-1]
    right = f[:, 1:]

    left_valid = ~jnp.isnan(left)
    right_valid = ~jnp.isnan(right)

    # Open boundary condition for NaN values:
    # - Both valid: average
    # - One valid: use the valid one
    # - Both NaN: NaN
    mid = jnp.where(
        left_valid & right_valid,
        (left + right) * 0.5,
        jnp.where(left_valid, left, jnp.where(right_valid, right, jnp.nan))
    )

    # Pad at the domain boundary with edge value (open boundary condition)
    mid = lax.cond(
        padding == "left",
        lambda: jnp.pad(mid, ((0, 0), (1, 0)), mode='edge'),
        lambda: jnp.pad(mid, ((0, 0), (0, 1)), mode='edge')
    )

    mid = jnp.moveaxis(mid, -1, axis)

    if land_mask is not None:
        mid = jnp.where(land_mask, jnp.nan, mid)

    return mid


def derivative(
    field: Float[jax.Array, "y x"],
    axis: Literal[0, 1],
    land_mask: Float[jax.Array, "y x"] = None,
) -> Float[jax.Array, "y x"]:
    """
    Differentiates a ``field``, using finite differences, along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/'x'),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain.

    An open boundary condition is applied (zero second derivative):

    - At domain edges: the boundary derivative equals the nearest interior derivative
    - At land/NaN boundaries: if one of the two values is NaN, the derivative is filled
      with the immediate neighbor derivative; if both neighbors are NaN, the result is NaN

    This is appropriate for domains with sharp physical boundaries (e.g., SWOT swaths)
    where the signal continues smoothly beyond the observation edge.

    Parameters
    ----------
    field : Float[jax.Array, "y x"]
        Field to differentiate
    axis : Literal[0, 1]
        Axis along which interpolation is performed
    land_mask : Float[jax.Array, "y x"], optional
        Mask indicating the land domain where extrapolation should be applied.
        `False`/`0` indicates ocean cells, `True`/`1` indicates land cells.

        Defaults to `None`, 
        in which case no land masking is applied and extrapolation is performed across the entire domain

    Returns
    -------
    df : Float[jax.Array, "y x"]
        Differentiated field
    """
    f = jnp.moveaxis(field, axis, -1)

    df = jnp.diff(f, axis=-1)

    # Open boundary condition for NaN values:
    # Fill NaN derivatives with immediate neighbor
    left_neighbor = jnp.roll(df, 1, axis=-1).at[..., 0].set(jnp.nan)
    right_neighbor = jnp.roll(df, -1, axis=-1).at[..., -1].set(jnp.nan)

    df = jnp.where(
        ~jnp.isnan(df),
        df,
        jnp.where(
            ~jnp.isnan(left_neighbor), left_neighbor,
            jnp.where(~jnp.isnan(right_neighbor), right_neighbor, jnp.nan)
        )
    )

    # Open boundary condition at domain edges: ∂²f/∂x² = 0
    # The boundary derivative equals the nearest interior derivative
    df = jnp.concatenate([df, df[..., -1:]], axis=-1)

    df = jnp.moveaxis(df, -1, axis)

    if land_mask is not None:
        df = jnp.where(land_mask, jnp.nan, df)

    return df


def horizontal_derivatives(
    field: Float[jax.Array, "y x"],
    lat: Float[jax.Array, "y x"] = None,
    lon: Float[jax.Array, "y x"] = None,
    dx: Float[jax.Array, "y x"] = None,
    dy: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Computes the horizontal derivatives of a ``field`` defined on an orthogonal grid, 
    using finite differences and applying an open boundary condition at domain edges and land/NaN boundaries.

    Horizontal derivatives are returned on the same grid as the input field, and in the grid coordinates.

    Parameters
    ----------
    field : Float[jax.Array, "y x"]
        Field for which to compute gradients
    lat : Float[jax.Array, "y x"], optional
        Latitude grid corresponding to the field

        Defaults to `None`, in which case ``dx`` and ``dy`` must be provided
    lon : Float[jax.Array, "y x"], optional
        Longitude grid corresponding to the field

        Defaults to `None`, in which case ``dx`` and ``dy`` must be provided
    dx : Float[jax.Array, "y x"], optional
        Grid spacing in the eastward direction (i.e. along axis=1)

        Defaults to `None`, in which case ``lat`` and ``lon`` must be provided
    dy : Float[jax.Array, "y x"], optional
        Grid spacing in the northward direction (i.e. along axis=1)

        Defaults to `None`, in which case ``lat`` and ``lon`` must be provided
    land_mask : Float[jax.Array, "y x"], optional
        Mask indicating the land domain where extrapolation should be applied.
        `False`/`0` indicates ocean cells, `True`/`1` indicates land cells.

        Defaults to `None`, 
        in which case no land masking is applied and extrapolation is performed across the entire domain
    
    Returns
    -------
    df_e : Float[jax.Array, "y x"]
        Eastward derivative of the field, on the same grid as the input field
    df_n : Float[jax.Array, "y x"]
        Northward derivative of the field, on the same grid as the input field
    """
    if dx is None or dy is None:
        if lat is None or lon is None:
            raise ValueError("Either lat/lon or dx/dy must be provided")
        from .geometry import grid_spacing
        dx, dy = grid_spacing(lat, lon)

    # compute derivatives in grid coordinates
    df_x = derivative(field, axis=1, land_mask=land_mask)
    df_y = derivative(field, axis=0, land_mask=land_mask)

    # interpolate from staggered grids to the T grid of the input field
    df_x = interpolation(df_x, axis=1, padding="left", land_mask=land_mask)
    df_y = interpolation(df_y, axis=0, padding="left", land_mask=land_mask)

    # scale derivatives with grid spacing to get physical derivatives
    df_x /= dx
    df_y /= dy

    return df_x, df_y
