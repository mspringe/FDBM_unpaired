r"""
This module provides a lightweight implementation of loss functions and update
steps.


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from warnings import warn
from functools import partial
import jax
from jax import value_and_grad, jit, pmap
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.lax import select
from flax.jax_utils import replicate
import optax
import jmp
import math
from .. import distributed
from ..model_zoo import predict
from .. import mixed_precision as mp
from ..jax_typing import PyTree, LossScale
from typing import Callable, List, Any, Tuple
from flax.training.train_state import TrainState


def map_dtype(
    tree: PyTree[jax.Array],
    dtype: jnp.dtype
) -> PyTree[jax.Array]:
    """
    this helper function enforces consient data types accross the PyTree

    Args:
        tree (PyTree): tree of e.g. parameters or optimizer states
        dtype (jnp.dtype): data type to enforce

    Returns:
        (PyTree) tree with enforced data type
    """
    def conditional_cast(x):
        is_instance = isinstance(x, jnp.ndarray)
        is_subtype = jnp.issubdtype(x.dtype, jnp.floating)
        if is_instance and is_subtype:
            x = x.astype(dtype)
        return x
    return jtu.tree_map(conditional_cast, tree)


def map_policy(policy_fn: Callable, *args: List[Any]) -> List[Any]:
    """
    maps the same mixed precision policy to all arguments
    
    Args:
        policy_fn (Callable): a policy to apply to all arguments
        args (List): arguments that should adhere to policy

    Returns:
        (List) arguments with enforced policy
    """
    return [policy_fn(a) for a in args]


def select_opt_state(
    condition: PyTree[jax.Array],
    o_true: PyTree[jax.Array], 
    o_false: PyTree[jax.Array]
) -> PyTree[jax.Array]:
    """
    this helper function allows you to select between PyTree states given some
    precalculated conditions

    Args:
        condition (PyTree): tree of conditions indicating what values to select
            at which nodes
        o_true (PyTree): values to select for true conditions
        o_false (PyTree): values to select for false conditions

    Returns:
        (PyTree) the resulting tree merged conditionally from `o_true` and 
        `o_false`
    """
    assert condition.ndim == 0 and condition.dtype == jnp.bool_, \
           "expected boolean scalar"
    slct_fn = partial(select, condition)
    return jtu.tree_map(slct_fn,
                        map_dtype(o_true, jnp.float32),
                        map_dtype(o_false, jnp.float32))


def safe_update(
    state: TrainState,
    grads: PyTree[jax.Array],
    loss_scaler: LossScale,
    skip_nans: bool = True
) -> Tuple[TrainState, LossScale]:
    """
    This method offers save updates, where nans will be avoided


    Args:
        state (TrainState): current training state
        grads (PyTree): gradients from from loss
        loss_scaler (LossScale): loss scaler
        skip_nans (bool): If true, nans will be avoided and respective
            optimization steps will be skipped

    Returns:
        (Tuple[TrainState, LossScale]) the updated training state and 
        loss Scaler
    """
    def apply_optimizer(params, grads):
        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
    grads = loss_scaler.unscale(grads)
    # You should put gradient clipping etc after unscaling.
    if skip_nans:
        grads_finite = jmp.all_finite(grads)
        # The loss scale will be periodically increased if gradients remain
        # finite and will be decreased if not.
        loss_scaler = loss_scaler.adjust(grads_finite)
        # select updates only if gradients are finite
        new_params, new_opt_state = apply_optimizer(state.params, grads)
        params = jmp.select_tree(grads_finite, new_params, state.params)
        opt_state = select_opt_state(grads_finite, new_opt_state, state.opt_state)
        # Update TrainState
        state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
        )
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss_scaler


def mae(apply_fn, params, x, t, directions, tgt, rngs_dropout, eps=None,
        mp_policy=mp.DEFAULT_POLICY, loss_scaler=mp.DEFAULT_LOSS_SCALER,
        karras_pred=False):
    r"""
    Args:
        eps: entropic regularization parameter, if specified Karras-like
             scaling will be applied (see Appendix J of Bortolli et al.)
    Returns:
        tuple of reduced loss and respective gradients
    """
    # rngs & enforce training mode
    kwargs = dict(rngs={'dropout': rngs_dropout}, train=True)
    # enforce compute policy
    args = map_policy(
        mp_policy.cast_to_compute, x, t, directions,
    )
    tgt = mp_policy.cast_to_output(tgt)
    # loss function
    @value_and_grad
    def loss(parameters):
        if karras_pred:
            pred = predict(t, eps, apply_fn, parameters, *args, **kwargs)
        else:
            pred = apply_fn(parameters, *args, **kwargs)
        pred = mp_policy.cast_to_output(pred)
        l = jnp.mean(jnp.absolute(pred-tgt))
        l = loss_scaler.scale(l)
        return l
    # calculate loss
    value, grad = loss(params)
    # enforce gradient parameter policy
    grad = mp_policy.cast_to_param(grad)
    return value, grad


def mse(apply_fn, params, x, t, directions, tgt, rngs_dropout, eps=None,
        mp_policy=mp.DEFAULT_POLICY, loss_scaler=mp.DEFAULT_LOSS_SCALER,
        karras_pred=False):
    r"""
    Args:
        eps: entropic regularization parameter, if specified Karras-like 
             scaling will be applied (see Appendix J of Bortolli et al.)
    Returns:
        tuple of reduced loss and respective gradients
    """
    # rngs & enforce training mode
    kwargs = dict(rngs={'dropout': rngs_dropout}, train=True)
    # enforce compute policy
    args = map_policy(
        mp_policy.cast_to_compute, x, t, directions,
    )
    tgt = mp_policy.cast_to_output(tgt)
    # loss function
    @value_and_grad
    def loss(parameters):
        if karras_pred:
            pred = predict(t, eps, apply_fn, parameters, *args, **kwargs)
        else:
            pred = apply_fn(parameters, *args, **kwargs)
        pred = mp_policy.cast_to_output(pred)
        l = jnp.mean((pred-tgt)**2)
        l = loss_scaler.scale(l)
        return l
    # calculate loss
    value, grad = loss(params)
    # enforce gradient parameter policy
    grad = mp_policy.cast_to_param(grad)
    return value, grad


def pseudo_huber(apply_fn, params, x, t, directions, tgt, rngs_dropout, c=None,
                 eps=None, mp_policy=mp.DEFAULT_POLICY,
                 loss_scaler=mp.DEFAULT_LOSS_SCALER, karras_pred=False):
    r"""
    Pseudo Huber loss, with automatic calculation of constant $c$
    (following Song et al. arxiv:2310.14189)

    Args:
        eps: entropic regularization parameter, if specified Karras-like 
             scaling will be applied (see Appendix J of Bortolli et al.)
    Returns:
        tuple of reduced loss and respective gradients
    """
    # rngs & enforce training mode
    kwargs = dict(rngs={'dropout': rngs_dropout}, train=True)
    # enforce compute policy
    args = map_policy(
        mp_policy.cast_to_compute, x, t, directions,
    )
    tgt = mp_policy.cast_to_output(tgt)
    if c is None:
        c = jnp.sqrt(jnp.prod(x.shape[1:])) * 0.00054
    # loss function
    @value_and_grad
    def loss(parameters):
        if karras_pred:
            pred = predict(t, eps, apply_fn, parameters, *args, **kwargs)
        else:
            pred = apply_fn(parameters, *args, **kwargs)
        pred = mp_policy.cast_to_output(pred)
        l = jnp.mean(jnp.sqrt((pred-tgt)**2 + c**2) - c)
        l = loss_scaler.scale(l)
        return l
    # calculate loss
    value, grad = loss(params)
    # enforce gradient parameter policy
    grad = mp_policy.cast_to_param(grad)
    return value, grad


@partial(jit, static_argnames=('take_step', 'loss_fn', 'mp_policy',
                               'karras_pred'))
def step(
        state,
        x,
        target,
        t,
        directions,
        rngs_dropout,
        take_step=True,
        loss_fn=mse,
        eps=None,
        skip_nans=True,
        mp_policy=mp.DEFAULT_POLICY,
        loss_scaler=mp.DEFAULT_LOSS_SCALER,
        karras_pred=False
    ):
    """
    Args:
        state: a `flax.training.train_state.TrainState` of respective model
        x: inputs
        target: target outputs
        t: time of respective direction (fwd / bwd)
        directions: encoding representing SDE-direction of respective network
                    (fwd / bwd)
        rngs_dropout: the PRNG (-key) for dropout operations
        take_step: only if this flag is set, an optimization step will be taken
        loss_fn: loss function (e.g. `mse`, `mae`, `pseudo_huber`)
        eps: entropic regularization parameter, if specified Karras-like
             scaling will be applied (see Appendix J of Bortolli et al.)
    Returns:
        tuple containing the loss, new training state and gradient scaling
        factor
    """
    loss, grads = loss_fn(
        state.apply_fn, state.params, x, t, directions, target, rngs_dropout,
        eps=eps, mp_policy=mp_policy, loss_scaler=loss_scaler,
        karras_pred=karras_pred
    )
    loss = loss_scaler.unscale(loss)
    # optimizer step
    if take_step:
        state, loss_scaler = safe_update(state, grads, loss_scaler, skip_nans)
    return loss, state, grads, loss_scaler
