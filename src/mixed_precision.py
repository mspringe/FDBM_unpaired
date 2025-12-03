"""
This module provides some helper functions for mixed precision training

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax.numpy as jnp
import jmp


DEFAULT_LOSS_SCALER = jmp.NoOpLossScale()
DEFAULT_POLICY = jmp.Policy(compute_dtype=jnp.float32,
                            param_dtype=jnp.float32,
                            output_dtype=jnp.float32)


def loss_scaling(
    precision: jnp.dtype,
    init_scale: float = 2**15,
    min_loss_scale: float = 1.0
) -> jmp.LossScale:
    """
    This method selects a loss scaler, depending on the precision used during
    training.

    Args:
        precision (jnp.dtype): precision of weights
        init_scale (float): initial scale for dynamic loss scaling
        min_loss_scale (float): minimum scale for dynamic loss scaling

    Returns:
        (jmp.LossScale) a loss-scaling
    """
    # no loss scaling for float32
    if precision == jnp.float32:
        return jmp.NoOpLossScale()
    # dynamic loss scaling otherwise
    return jmp.DynamicLossScale(
        loss_scale=jnp.float32(init_scale),
        min_loss_scale=jnp.float32(min_loss_scale)
    )


def policy(precision: jnp.dtype):
    """mixed precision training policy, where weights are stored as FP32"""
    return jmp.Policy(compute_dtype=precision,
                      param_dtype=jnp.float32,
                      output_dtype=precision)
