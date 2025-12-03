"""
This module contains all helper functions w.r.t. the loss of FDBM models for
unpaired data.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from .fsb import input_transform, cond_var
from functools import partial
from jax import jit, Array


@jit
def make_target(
    x: Array,
    xt: Array,
    yt: Array,
    t: Array,
    T: float,
    omega: Array,
    gamma: Array,
    g: float,
    direction: int = 1
) -> Array:
    """
    Args:
        x (jax.Array): Target samplefrom terminal distribution
        xt (jax.Array): X-elements of interpolated sample from the bridge
        yt (jax.Array): Y-elements of interpolated sample from the bridge
        t (jax.Array): points in time of shape (batch_size,)
        T (float): maximium point in time, e.g. 1.0
        omega (jax.Array): MAfBM weights
        gamma (jax.Array): MAfBM geometric grid
        g (float): value of diffusion function g at t 
            (e.g. constant std of entropic reg.)
        direction (int): directions of shape (K,), 
            where 1 -> forward and 0 -> backward

    Returns:
        (jax.Array) the target for the neural network
    """
    # sigma_{T|t}
    sigma_Tt = cond_var(t,T,omega,gamma,g)
    # shape sanity
    while len(sigma_Tt.shape) != len(x.shape):
        sigma_Tt = sigma_Tt[...,None]
    # input transform zt ->  xt_in
    xt_in = input_transform(xt, yt, t, T, omega, gamma, g, direction=direction)
    return (x - xt_in) / sigma_Tt
