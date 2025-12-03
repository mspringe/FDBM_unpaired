"""
This module provides low level functionalities to calculate the
parameterization of MAfBM, given a Hurst-Index H and a number of additional
(augmenting) processes K.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import math
import jax
import jax.numpy as jnp
import jax.scipy as jscp
from functools import partial
from jax import Array
from typing import Tuple


def gamma_by_r(K: int, r: float, offset: float = 0.) -> Array:
    """
    This method sets up the geometrically spaced grid of quadrature values for
    the speeds of mean reversion of augmenting processes.
    $$
        gamma_k = r^{k - (K+1)/2 - offset}
    $$

    Args:
        K (int): total number of augmenting processes
        r (int): base of exponential spacing
        offset (float): optional offset for exponent, defaults to zero

    Returns:
        (jax.Array) geometrically spaced grid `gamma` for the speeds of mean
        reversion of augmenting processes
    """
    n = (K+1) / 2 + offset
    ks = jnp.arange(1, K+1)
    gamma = r ** (ks - n)
    return gamma


def gamma_by_gamma_max(K: int, gamma_max: float, offset: float = 0.) -> Array:
    """
    This method sets up the geometrically spaced grid for augmenting processes.
    Here we derive the grid directly from the maximum gamma value
    $$
        r = \gamma_{\max} ^ {K - 1 - 2*offset}.
    $$

    See `gamma_by_r` for how we construct the grid.

    Args:
        K (int): total number of augmenting processes
        gamma_max (int): maximum gamma value
        offset (float): optional offset for exponent, defaults to zero

    Returns:
        (jax.Array) geometrically spaced grid `gamma` for the speeds of mean
        reversion of augmenting processes
    """
    r = gamma_max ** (2 / (K - 1 - 2 * offset))
    return gamma_by_r(K, r, offset)


def gamma_by_range(K: int, gamma_min: float, gamma_max: float) -> Array:
    """
    Alternatively, one could construct the grid using simpler methods, such as
    $\gamma_k = e^{s}$, where we take equidistant steps w.r.t. s in 
    [log(gamma_min), log(gamma_max)].

    Args:
        K (int): total number of augmenting processes
        gamma_max (int): minimum gamma value
        gamma_max (int): maximum gamma value

    Returns:
        (jax.Array) geometrically spaced grid `gamma` for the speeds of mean
        reversion of augmenting processes
    """
    return jnp.exp(jnp.linspace(jnp.log(gamma_min), jnp.log(gamma_max), K))


def calc_gamma(K: int, gamma_max: float, gamma_min: float = None) -> Array:
    """
    This method chooses between `gamma_by_gamma_max` and `gamma_by_range` 
    depending on the inputs.

    If you do not specify anny `gamma_min`, the method will default to an 
    'gamma_by_gamma_max'.

    Args:
        K (int): total number of augmenting processes
        gamma_max (int): minimum gamma value
        gamma_max (int): maximum gamma value

    Returns:
        (jax.Array) geometrically spaced grid `gamma` for the speeds of mean
        reversion of augmenting processes
    """
    if K == 1:
        return gamma_by_r(K, math.sqrt(gamma_max))
    if gamma_min is None:
        return gamma_by_gamma_max(K, gamma_max)
    gamma = gamma_by_range(K, gamma_min, gamma_max)
    return gamma


def calc_omega(
    gamma: Array,
    H: float,
    T: float = 1.0,
    return_cost: bool = False,
    return_Ab : bool = False
) -> Tuple[Array, ...]:
    """
    Calculation of approximation coefficients for MAfBM, based on the mean
    approximation error with type II fractional brownian motion.

    Args:
        gamma (jax.Array): speed of mean reversion
        H (float): hurst-index
        T (float): time-horizon / largest time-step that is considered
        return_cost (bool): if specified, the cost of the approximation is returned as
            well
        return_Ab (bool): if set to true, the linear equation system will
            be returned as well

    Return:
        Approximation coefficients of MAfBM, needed to approximate
        fractional brownian motion in markovian setting.
    """
    if not isinstance(T, float):
        T = float(T.squeeze())
    # FP64 for higher precision of inverse matrix
    jax.config.update("jax_enable_x64", True)
    gamma = gamma.astype(jnp.float64)
    if not isinstance(H, float):
        H = H.astype(jnp.float64)
    # formula
    gamma_i, gamma_j = gamma[None, :], gamma[:, None]
    mat_gamma = gamma_i + gamma_j
    # setting up A
    exponential = jnp.exp( -mat_gamma * T )
    A = (T + (exponential - 1) / mat_gamma) / mat_gamma
    # setting up b
    lowH, highH = H+0.5, H+1.5
    gammainc_lowH = jscp.special.gammainc(lowH, gamma*T)
    gammainc_highH = jscp.special.gammainc(highH, gamma*T)
    b = T / gamma**lowH * gammainc_lowH - lowH / gamma**highH * gammainc_highH
    # solve the linear programm
    # FP64 for higher precision of inverse matrix
    omega = jscp.linalg.solve(A.astype(jnp.float64), b.astype(jnp.float64))
    # calculate the cost if specified
    if return_cost:
        exponential = jnp.exp(jscp.special.gammaln(H + 0.5))
        c = T**(2 * H + 1) / (2 * H) / (2 * H + 1) / exponential ** 2
        cost = 1 - b @ omega / c
    # gather outputs into a single tuple
    # Output can be FP32
    if not return_Ab:
        output = omega.astype(jnp.float32) 
    else:
        output = (
            omega.astype(jnp.float32), 
            A.astype(jnp.float32),
            b.astype(jnp.float32)
        )
    output = output if not return_cost else (*output, cost.astype(jnp.float32))
    return output


def calc_omega_gamma(
    H: float,
    K: int, 
    gamma_max: float = 40.0,
    gamma_min: float = 0.1,
    T: float = 1.0
) -> Tuple[Array, Array]:
    """
    This helper method calculates and returnes both omega and gamma for MA-fBM.

    Args:
        H (float): hurst-index
        K (int): total number of augmenting processes
        gamma_max (int): minimum gamma value
        gamma_max (int): maximum gamma value
        T (float): time-horizon / largest time-step that is considered

    Return:
        Approximation coefficients `omega` and geometrically spaced grid 
        `gamma` of MAfBM.
    """
    if K > 0:
        gamma = calc_gamma(K, gamma_max, gamma_min)
        omega = calc_omega(gamma, H, T)
    else:
        gamma = jnp.ones([1]) * 0
        omega = jnp.ones([1]) * 1
    return omega, gamma
