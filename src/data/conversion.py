"""
This module provides simple helper functions to map between numpy UInt8 data
and jax tensors spanning arbitrary intervals and data types.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import numpy as np
import jax.numpy as jnp
from jax import Array


def scale_vision(
    arr: Array,
    max_val_in: float = 1.0,
    min_val_in: float = 0.0,
    dtype: jnp.dtype = jnp.float32
) -> Array:
    """
    scales inputs to the interval [-1, 1]

    Args:
        arr (jax.Array): arbitrary array-like (e.g. jax.numpy) to be scaled
        max_val_in (float): maximum or supremum for the domain of `arr`
        min_val_in (float): minimum or infimum for the domain of `arr`
        dtype (jnp.dtype):  the preferred floating-point datatype for outputs
            compute will always be carried out in Float32

    Returns:
        values of `arr` linearly scaled to [-1, 1]
    """
    delta = max_val_in - min_val_in
    scale = delta / 2
    arr = arr.astype(jnp.float32)
    arr = (arr - min_val_in) / scale - 1
    arr = arr.astype(dtype)
    return arr


def unscale_vision(
    arr: Array,
    max_val_out: float = 1.0,
    min_val_out: float = 0.0,
    dtype: jnp.dtype = jnp.float32
) -> Array:
    """
    scales inputs from [-1, 1] to [`max_val_out`, `min_val_out`]

    Args:
        arr: arbitrary array-like (e.g. jax.numpy) in [-1, 1]
        max_val_out: maximum for the output domain
        min_val_out: minimum for the output domain
        dtype: (optional) the preferred floating-point datatype for outputs
               compute will always be carried out in Float32
    Returns:
        values of `arr` linearly scaled to [`max_val_out`, `min_val_out`]
    """
    delta = max_val_out - min_val_out
    arr = arr.astype(jnp.float32)
    arr = (arr + 1) * (delta/2) + min_val_out  # values in [min, max]
    arr = arr.astype(dtype)
    return arr


def to_npuint8(
    arr: Array,
    max_val_in: float = 255.0,
    min_val_in: float = 0.0
) -> Array:
    """
    scales and converts input to numpy UInt8 datatype

    Args:
        arr: arbitrary array-like (e.g. jax.numpy) to be converted
        max_val_in: maximum or supremum for the domain of `arr`
        min_val_in: minimum or infimum for the domain of `arr`
        dtype: (optional) the preferred floating-point datatype for outputs
               compute will always be carried out in Float32
    Returns:
        values of `arr` linearly scaled to [-1, 1]
    """
    delta = max_val_in - min_val_in
    scale = 255. / delta
    arr = arr.astype(jnp.float32)
    arr = (arr - min_val_in) * scale
    arr = jnp.round(arr)  # important: rounding before converting to UInt8
    arr = jnp.clip(arr, 0, 255).astype(jnp.uint8)
    arr = np.array(arr).astype(np.uint8)
    return arr


def from_npuint8(
    arr: Array,
    max_val_out: float = 1.0,
    min_val_out: float = 0.0,
    dtype: jnp.dtype = jnp.float32
):
    """
    scales and convert input from numpy UInt8 to specified floating points

    Args:
        arr: arbitrary array-like (e.g. jax.numpy) to be converted
        max_val_out: maximum for the output domain
        min_val_out: minimum for the output domain
        dtype: (optional) the preferred jax floating-point datatype for outputs
               compute will always be carried out in Float32
    Returns:
        values of `arr` linearly scaled to [-1, 1]
    """
    delta = max_val_out - min_val_out
    scale = delta / 255.
    arr = jnp.array(arr).astype(jnp.float32)
    arr = arr * scale + min_val_out
    arr = arr.astype(dtype)
    return arr
