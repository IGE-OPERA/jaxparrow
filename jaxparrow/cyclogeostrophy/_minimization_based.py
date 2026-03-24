import inspect
from functools import partial
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float
import optax

from ._core import CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _cyclogeostrophic_loss
from ..utils import operators
from ..utils.geometry import GRAVITY


_SYSTEM_PARAMS = frozenset({
    "ssh_t", "ug_t", "vg_t", "ucg_t", "vcg_t",
    "lat_t", "lon_t",
    "dx_e_t", "dx_n_t", "dy_e_t", "dy_n_t", "J_t",
    "coriolis_factor_t", "land_mask",
})


def minimization_based(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
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

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``mask``). Geostrophic velocities are provided on the T grid
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
        Its signature determines its behavior:

        - Parameter names from ``{ssh_t, ug_t, vg_t, ucg_t, vcg_t, lat_t, lon_t,
          dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask}``
          are automatically provided.
        - Any other parameter names must be provided via ``reg_kwargs``.
        - If ``ssh_t`` is in the signature, the SSH field becomes an optimized parameter.
          Gradients from the cyclogeostrophic imbalance are stopped w.r.t. SSH
          (via ``jax.lax.stop_gradient``), so only the regularization term drives SSH updates.
          Geostrophic velocities are recomputed from the (evolving) SSH at each iteration.
          If ``ug_t``/``vg_t`` are also requested, they reflect the current SSH-derived values.
        - If ``ug_t`` and/or ``vg_t`` are in the signature (without ``ssh_t``),
          the geostrophic velocities become optimized parameters.
          Gradients from the cyclogeostrophic imbalance are stopped w.r.t. ``ug``/``vg``,
          so only the regularization term drives their updates.

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
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask
    )

    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": 0.005}
        optim = getattr(optax, optim)(**optim_kwargs)
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError(
            "optim should be an optax.GradientTransformation optimizer, or a string referring to such an optimizer."
        )

    # Handle regularization and SSH/geos optimization
    reg_wrapper = None
    optimize_ssh = False
    optimize_geos = False
    if regularization is not None:
        sig = inspect.signature(regularization)
        reg_params = set(sig.parameters)
        optimize_ssh = "ssh_t" in reg_params
        optimize_geos = bool(reg_params & {"ug_t", "vg_t"}) and not optimize_ssh
        if optimize_ssh and ssh_t is None:
            raise ValueError(
                "Regularization function requests ssh_t but no SSH field was provided. "
                "Provide ssh_t to enable SSH optimization."
            )
        reg_wrapper = _build_reg_wrapper(regularization, reg_kwargs)

    ucg, vcg, opt_ssh, opt_ug, opt_vg, losses = _minimization_based(
        setup.ug_t, setup.vg_t,
        setup.dx_e_t, setup.dx_n_t, setup.dy_e_t, setup.dy_n_t, setup.J_t,
        setup.coriolis_factor_t,
        setup.land_mask, n_it, optim,
        regularization=reg_wrapper, lat_t=lat_t, lon_t=lon_t,
        ssh_t=ssh_t if optimize_ssh else None,
        reg_kwargs=reg_kwargs,
        optimize_ssh=optimize_ssh, optimize_geos=optimize_geos,
    )
    return assemble_result(
        ucg, vcg, setup,
        return_geos=return_geos, return_losses=return_losses, losses=losses,
        ssh_t=opt_ssh, ug_t=opt_ug, vg_t=opt_vg,
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

    def wrapper(ucg_t, vcg_t, ssh_t, ug_t, vg_t, lat_t, lon_t,
                dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask,
                reg_kwargs):
        all_system = {
            "ssh_t": ssh_t, "ug_t": ug_t, "vg_t": vg_t,
            "ucg_t": ucg_t, "vcg_t": vcg_t,
            "lat_t": lat_t, "lon_t": lon_t,
            "dx_e_t": dx_e_t, "dx_n_t": dx_n_t,
            "dy_e_t": dy_e_t, "dy_n_t": dy_n_t,
            "J_t": J_t, "coriolis_factor_t": coriolis_factor_t, "land_mask": land_mask,
        }
        kwargs = {}
        for name in param_names:
            if name in _SYSTEM_PARAMS:
                kwargs[name] = all_system[name]
            else:
                kwargs[name] = reg_kwargs[name]
        return regularization(**kwargs)

    return wrapper


@partial(jax.jit, static_argnames=("n_it", "optim", "regularization", "optimize_ssh", "optimize_geos"))
def _minimization_based(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
    n_it: int,
    optim: optax.GradientTransformation,
    regularization: Callable = None,
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    ssh_t: Float[jax.Array, "y x"] = None,
    reg_kwargs: dict = None,
    optimize_ssh: bool = False,
    optimize_geos: bool = False,
):
    def loss_fn(args):
        if optimize_ssh:
            ucg, vcg, ssh = args
            # Recompute geostrophic velocities from SSH;
            ug_raw, vg_raw = _compute_geostrophy_from_grid_metrics(
                ssh, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask, coriolis_factor_t
            )
            # stop_gradient on the result so only the regularization drives SSH updates
            ug = jax.lax.stop_gradient(ug_raw)
            vg = jax.lax.stop_gradient(vg_raw)
            ug_reg, vg_reg = ug_raw, vg_raw
        elif optimize_geos:
            ucg, vcg, ug_opt, vg_opt = args
            # stop_gradient so only the regularization drives ug/vg updates
            ug = jax.lax.stop_gradient(ug_opt)
            vg = jax.lax.stop_gradient(vg_opt)
            ug_reg, vg_reg = ug_opt, vg_opt
            ssh = None
        else:
            ucg, vcg = args
            ug, vg = ug_t, vg_t
            ug_reg, vg_reg = ug_t, vg_t
            ssh = None

        loss = _cyclogeostrophic_loss(
            ug, vg, ucg, vcg, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask
        )

        if regularization is not None:
            loss += regularization(
                ucg, vcg, ssh, ug_reg, vg_reg, lat_t, lon_t,
                dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask,
                reg_kwargs,
            )

        return loss

    if optimize_ssh:
        init_params = (ug_t, vg_t, ssh_t)
    elif optimize_geos:
        init_params = (ug_t, vg_t, ug_t, vg_t)
    else:
        init_params = (ug_t, vg_t)

    def step_fn(carry, _):
        params = carry[:-1]
        opt_state = carry[-1]

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = tuple(map(lambda x: jnp.nan_to_num(x, copy=False, nan=0, posinf=0, neginf=0), grads))

        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params + (opt_state,), loss

    carry, losses = lax.scan(
        step_fn, init_params + (optim.init(init_params),), xs=None, length=n_it
    )

    if optimize_ssh:
        ucg, vcg, ssh = carry[:-1]
        return ucg, vcg, ssh, None, None, losses
    elif optimize_geos:
        ucg, vcg, ug, vg = carry[:-1]
        return ucg, vcg, None, ug, vg, losses
    else:
        ucg, vcg = carry[:-1]
        return ucg, vcg, None, None, None, losses


def _compute_geostrophy_from_grid_metrics(ssh, dx_e, dx_n, dy_e, dy_n, J, land_mask, coriolis_factor):
    deta_e, deta_n = operators.horizontal_derivatives(
        ssh, dx_e=dx_e, dx_n=dx_n,
        dy_e=dy_e, dy_n=dy_n, J=J, land_mask=land_mask
    )
    ug = -GRAVITY * deta_n / coriolis_factor
    vg = GRAVITY * deta_e / coriolis_factor
    return ug, vg
