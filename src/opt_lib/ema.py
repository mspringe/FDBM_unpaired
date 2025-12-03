"""
This module provides a lightweight implementation of EMA updates


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from functools import partial
import jax
import jax.tree_util as jtu
from ..jax_typing import PyTree


@partial(jax.jit, static_argnames=['ema_rate'])
def ema_operation(
    ema_element: jax.Array,
    model_element: jax.Array,
    ema_rate: float
) -> jax.Array:
    """
    jit compiled EMA operation, can be mapped to a PyTree.

    Args:
        ema_element (jax.Array): Parameter from the EMA model
        model_element (jax.Array): Parameter from the online model
        ema_rate (float): Exponential decay rate

    Returns:
        Updated EMA parameter
    """
    return ema_rate * ema_element + (1-ema_rate) * model_element


@partial(jax.jit, static_argnames=['ema_rate'])
def update_ema(
    ema_parameters: PyTree[jax.Array],
    model_parameters: PyTree[jax.Array],
    ema_rate: float
) -> PyTree[jax.Array]:
    """
    efficient omputation of all EMA updates w.r.t. some EMA of parameters and
    parameters of an online model

    Args:
        ema_parameters (PyTree): EMA parameters
        model_element (dict): Online model's parameters
        ema_rate (float): Exponential decay rate

    Returns:
        Updated EMA parameters
    """
    ema_fn = partial(ema_operation, ema_rate=ema_rate)
    new_params = jtu.tree_map(ema_fn, ema_parameters, model_parameters)
    return new_params
