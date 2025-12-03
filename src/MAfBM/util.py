"""
minor helper functions

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from jax import Array
import jax.numpy as jnp
from einops import rearrange


def squeeze(arr: Array) -> Array:
    """
    squeezes array, but does not convert to scalar values (at least 1d)

    Args:
        arr: array to squeeze

    Return:
        (jax.Array) squeezed array
    """
    arr = arr.squeeze()
    if len(arr.shape) == 0:
        arr = jnp.atleast_1d(arr)
    return arr


def matrix_vector_mp(A: Array, v: Array) -> Array:
    """
    Args:
        A (jax.Array): matrix of shape (BS, H, W, C, M, M)
        v (jax.Array): vector of shape (BS, H, W, C, M)
    """
    # Reshape v to have an additional dimension at the end 
    # (for matrix-vector multiplication)
    v_expanded = v[...,None]  # shape (BS, H, W, C, M, 1)
    # Perform matrix-vector multiplication
    result = A @ v_expanded
    # Shape (BS, H, W, C, M, 1), remove the last dimension
    result_squeezed = result.squeeze(-1)  # shape (BS, H, W, C, M)
    return result_squeezed


def flatten_Z(Z: Array) -> Array:
    """flattens Z such that all processes share the feature dimension"""
    return rearrange(Z, 'B H W C K -> B H W (C K)')


def unflatten_Z(Z: Array, num_channels: int = 3) -> Array:
    """reverses `flatten_Z`"""
    D = Z.shape[-1]
    c = num_channels
    k = D // c
    return rearrange(Z, 'B H W (C K) -> B H W C K', C=c, K=k)
