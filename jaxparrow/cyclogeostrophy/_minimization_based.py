import inspect
from functools import partial
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float
import optax

from ._core import CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _cyclogeostrophic_loss


_SYSTEM_PARAMS = frozenset({
    "ucg_t", "vcg_t", "lat_t", "lon_t",
    "dx_t", "dy_t", "coriolis_factor_t", "land_mask",
})


def minimization_based(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    is_grid_rectilinear: bool | None = None,
    rotate_to_geographic: bool = True,
    return_geos: bool = False,
    return_losses: bool = False,
    n_it: int = 2000,
    optim: optax.GradientTransformation | str = "sgd",
    optim_kwargs: dict = None,
    regularization: Callable = None,
    reg_kwargs: dict = None,
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using our minimization-based method.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``land_mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``land_mask``). Geostrophic velocities are provided on the T grid
       and will be interpolated to U/V grids internally.

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitude of the T grid.
    lon_t : Float[jax.Array, "y x"]
        Longitude of the T grid.
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field (on the T grid). Required if geostrophic velocities are not provided.
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid. If provided with ``vg_t``,
        bypasses SSH-based computation. Will be interpolated to U grid.
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid. If provided with ``ug_t``,
        bypasses SSH-based computation. Will be interpolated to V grid.
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` or ``ug_t`` `nan` values

        Defaults to `None`
    is_grid_rectilinear : bool, optional
        If `True`, the grid is assumed to be rectilinear in geographic coordinates.
        If `False`, the grid is assumed to be curvilinear and the grid angle is computed from the grid spacing. 
        If `None`, the grid is assumed to be rectilinear if the grid angle computed from the grid spacing is close to zero everywhere, and curvilinear otherwise.

        Defaults to `None`
    rotate_to_geographic : bool, optional
        If `True`, rotates the output velocities from grid-relative to geographic coordinates.
        Rotation is performed using the grid angle computed from the grid spacing.
        If `False`, output velocities are in grid-relative coordinates.

        If using a rectilinear grid in geographic coordinates, set to `False` to avoid unnecessary rotation.

        Defaults to `True`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`
    return_losses : bool, optional
        If `True`, returns the losses (cyclogeostrophic imbalance) over iterations.

        Defaults to `False`
    n_it : int, optional
        Maximum number of iterations.

        Defaults to `2000`
    optim : optax.GradientTransformation | str, optional
        Optimizer to use.
        Can be an ``optax.GradientTransformation`` optimizer, or a ``string`` referring to such an optimizer.

        Defaults to `sgd`
    optim_kwargs : dict, optional
        Optimizer arguments (such as learning rate, etc...).

        If `None`, only the learning rate is enforced to `0.005`

        Defaults to `None`
    regularization : Callable, optional
        A regularization function added to the cyclogeostrophic loss at every iteration.
        Its signature is defined as follows:

        - Parameter names from ``{ucg_t, vcg_t, lat_t, lon_t, dx_t, dy_t, coriolis_factor_t, land_mask}`` are automatically provided,
        but only ``ucg_t`` and ``vcg_t`` are required.
        - Any other parameter names must be provided via ``reg_kwargs``.

        Must return a scalar.

        Defaults to `None`
    reg_kwargs : dict, optional
        Additional keyword arguments passed to the ``regularization`` function.
        Values should be tracable JAX Pytrees.

        Defaults to `None`

    Returns
    -------
    CyclogeostrophyResult
        Named tuple containing:
        - ``ucg``: $u$ component of cyclogeostrophic velocity, on the T grid
        - ``vcg``: $v$ component of cyclogeostrophic velocity, on the T grid
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True`` or optimized via regularization)
        - ``ssh``: Optimized SSH field (if SSH regularization was used)
        - ``losses``: Cyclogeostrophic imbalance per iteration (if ``return_losses=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask, is_grid_rectilinear=is_grid_rectilinear
    )

    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": 0.005}
        optim = getattr(optax, optim)(**optim_kwargs)
        optim = optax.chain(optax.clip(1.0), optim)  # Clip gradients to prevent instability
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError(
            "optim should be an optax.GradientTransformation optimizer, or a string referring to such an optimizer."
        )

    # Handle regularization
    reg_wrapper = None
    if regularization is not None:
        reg_wrapper = _build_reg_wrapper(regularization, reg_kwargs)

    ucg, vcg, losses = _minimization_based(
        setup.ug_t, setup.vg_t,
        setup.dx_t, setup.dy_t,
        setup.coriolis_factor_t,
        setup.land_mask, n_it, optim,
        regularization=reg_wrapper, lat_t=lat_t, lon_t=lon_t,
        reg_kwargs=reg_kwargs,
    )

    return assemble_result(
        ucg, vcg, setup, rotate_to_geographic, return_geos, return_losses=return_losses, losses=losses,
    )


def _build_reg_wrapper(regularization, reg_kwargs):
    sig = inspect.signature(regularization)
    param_names = list(sig.parameters.keys())

    user_params = [n for n in param_names if n not in _SYSTEM_PARAMS]

    if user_params:
        if reg_kwargs is None:
            raise ValueError(
                f"Regularization function expects parameters {user_params} "
                f"that are not system parameters. Provide them via reg_kwargs."
            )
        missing = set(user_params) - set(reg_kwargs.keys())
        if missing:
            raise ValueError(
                f"Regularization function expects parameters {missing} "
                f"not found in reg_kwargs. Available reg_kwargs keys: {list(reg_kwargs.keys())}"
            )

    def wrapper(
        ucg_t, vcg_t, lat_t, lon_t,
        dx_t, dy_t, coriolis_factor_t, land_mask,
        reg_kwargs
    ):
        all_system = {
            "ucg_t": ucg_t, "vcg_t": vcg_t,
            "lat_t": lat_t, "lon_t": lon_t,
            "dx_t": dx_t, "dy_t": dy_t,
            "coriolis_factor_t": coriolis_factor_t, "land_mask": land_mask,
        }
        kwargs = {}
        for name in param_names:
            if name in _SYSTEM_PARAMS:
                kwargs[name] = all_system[name]
            else:
                kwargs[name] = reg_kwargs[name]
        return regularization(**kwargs)

    return wrapper


@partial(jax.jit, static_argnames=("n_it", "optim", "regularization"))
def _minimization_based(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_t: Float[jax.Array, "y x"],
    dy_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
    n_it: int,
    optim: optax.GradientTransformation,
    regularization: Callable = None,
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    reg_kwargs: dict = None,
):
    def loss_fn(args):
        ucg, vcg = args

        loss = _cyclogeostrophic_loss(ug_t, vg_t, ucg, vcg, dx_t, dy_t, coriolis_factor_t, land_mask)

        if regularization is not None:
            loss += regularization(
                ucg, vcg, lat_t, lon_t,
                dx_t, dy_t, coriolis_factor_t, land_mask,
                reg_kwargs,
            )

        return loss

    init_params = (ug_t, vg_t)

    def step_fn(carry, _):
        params = carry[:-1]
        opt_state = carry[-1]

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = tuple(map(lambda x: jnp.nan_to_num(x, copy=False, nan=0, posinf=0, neginf=0), grads))

        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params + (opt_state,), loss

    carry, losses = lax.scan(step_fn, init_params + (optim.init(init_params),), xs=None, length=n_it)

    ucg, vcg = carry[:-1]

    return ucg, vcg, losses
