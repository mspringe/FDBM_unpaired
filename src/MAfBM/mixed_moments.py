"""
This module provides low level functionalities to calculate all necessary
mixed moments (i.e., covariances) and means, required to apply MAfBM to various
SDE frameworks.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
import jax.numpy as jnp
import math
from functools import partial
from einops import rearrange
from .util import squeeze


def zeta(
    s: jax.Array,
    t: jax.Array,
    gamma: jax.Array,
    g: float
) -> jax.Array:
    """

    Args:
        s: time points of shape (batch_size,) and s<=t
        t: time points of shape (batch_size,) and s<=t
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)

    Returns:
        zeta weights of shape (B, 1, K)
    """
    if isinstance(s, float) or isinstance(s, int):
        s = jnp.array([s])
    if isinstance(t, float) or isinstance(t, int):
        t = jnp.array([t])
    # shape sanity
    s = rearrange(squeeze(s), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    gamma = rearrange(squeeze(gamma), 'K -> 1 1 K')
    # compute
    return g * (jnp.exp(-gamma * (t-s)) - 1)


def meanX(
    s: jax.Array,
    t: jax.Array,
    x: jax.Array,
    Y: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    keepdims: bool = False
) -> jax.Array:
    """
    compute E[X(t)|Z_s=z] with s<t, Z_s = (x,Y)

    Args:
        s: time points of shape (batch_size,) and s<t
        t: time points of shape (batch_size,) and s<t
        x: first element of Z_s
        Y: augmented processes, i.e. remaining elements of Z_s
        omega: MAfBM weights of shape (K,)
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)

    Returns:
        pinned mean E[X(t)|Z_s=z]
    """
    # shape sanity
    s = rearrange(squeeze(s), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    gamma = rearrange(squeeze(gamma), 'K -> 1 1 K')
    omega = rearrange(squeeze(omega), 'K -> 1 1 K')
    # compute
    weight = omega * zeta(s, t, gamma, g)
    y_part = jnp.sum(weight * Y, dim=-1, keepdims=keepdims)
    return x + y_part


def meanY(
    s: jax.Array,
    t: jax.Array,
    Y: jax.Array,
    gamma: jax.Array
) -> jax.Array:
    """
    compute E[Y(t)|Z_s=z) with s<t

    Args:
        s: time points of shape (batch_size,) and s<t
        t: time points of shape (batch_size,) and s<t
        Y: augmented processes, i.e. remaining elements of Z_s
        gamma: MAfBM weights of shape (K,)

    Returns:
        pinned mean E[Y(t)|Z_s=z]
    """
    # shape sanity
    s = rearrange(squeeze(s), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    gamma = rearrange(squeeze(gamma), 'K -> 1 1 K')
    # compute
    return jnp.exp( -gamma * (t-s) ) * Y


def meanZ(
    s: jax.Array,
    t: jax.Array,
    x: jax.Array,
    Y: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float
) -> jax.Array:
    """
    pinned mean of X and Y w.r.t. Z

    Args:
        s: time points of shape (batch_size,) and s<t
        t: time points of shape (batch_size,) and s<t
        x: first element of Z_s
        Y: augmented processes, i.e. remaining elements of Z_s
        omega: MAfBM weights of shape (K,)
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)

    Returns:
        pinned mean
    """
    mean_x = meanX(s, t, x, Y, omega, gamma, g, keepdims=True)
    mean_y = meanY(s, t, gamma)
    return jnp.cat([mean_x, mean_y], dim=-1)  # shape (B, 1, K+1)


@partial(jax.jit, static_argnames=('keepdims',))
def covX(
    s: jax.Array,
    t: jax.Array,
    T: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    keepdims: bool = False
) -> jax.Array:
    """
    compute cov(X(t), X(T) | Z_s=z) with s<t<=T

    Args:
        t: time points of shape (batch_size,)
        s: time points of shape (batch_size,)
        T: time-horizon / largest time-step that is considered (float)
        omega: MAfBM weights of shape (K,)
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)
        keepdims: flag indicating whether to keep the dimensions after sum

    Returns:
        cov(X(t), X(T) | Z_s=z)
    """
    # shape sanity
    if not (isinstance(T, float) or isinstance(T, int)):
        T = rearrange(squeeze(T), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    s = rearrange(squeeze(s), 'B -> B 1 1')
    omega = rearrange(squeeze(omega), 'K -> 1 K')
    gamma = rearrange(squeeze(gamma), 'K -> 1 K')
    # precomputing sub-terms
    omega_ij = omega[:,:,None] * omega[:,None,:]  # shape (1, K, K)
    gamma_i = gamma[:,:,None]                     # shape (1, K, 1)
    gamma_j = gamma[:,None,:]                     # shape (1, 1, K)
    gamma_ij =  gamma_i + gamma_j                 # shape (1, K, K)
    # calculating cov
    weight = omega_ij / gamma_ij
    exponentials = jnp.exp(-(T-t) * gamma_j) \
                   - jnp.exp(-(T-s) * gamma_j) * jnp.exp(-(t-s) * gamma_i)
    S = weight * exponentials                                 # shape (B, K, K)
    return g**2 * jnp.sum(S, axis=(1, 2), keepdims=keepdims)  # shape (B, 1, 1)


@partial(jax.jit, static_argnames=('keepdims',))
def covYX(
    t: jax.Array,
    T: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    keepdims: bool = False
) -> jax.Array:
    """
    computes covariance of Y processes and X

    Args:
        t: time points of shape (batch_size,) and s<t
        T: time-horizon / largest time-step that is considered (float)
        omega: MAfBM weights of shape (K,)
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)
        keepdims: flag indicating whether to keep the dimensions after sum

    Returns:
        covariance of Y with X
    """
    # shape sanity
    if not (isinstance(T, float) or isinstance(T, int)):
        T = rearrange(squeeze(T), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    gamma_l = rearrange(squeeze(gamma), 'K -> 1 K 1')
    gamma_k = rearrange(squeeze(gamma), 'K -> 1 1 K')
    omega_k = rearrange(squeeze(omega), 'K -> 1 1 K')
    # compute
    weight = omega_k / (gamma_l + gamma_k)                  # shape (1, K, K)
    exponentials = jnp.exp( -(T-t) * gamma_k ) \
                   - jnp.exp( -t * gamma_l - T * gamma_k )  # shape (B, K, K)
    S = weight * exponentials                               # shape (B, K, K)
    return g * jnp.sum(S, axis=2, keepdims=keepdims)        # shape (B, K, 1)


@jax.jit
def covY(
    s: jax.Array,
    t: jax.Array,
    T: jax.Array,
    gamma: jax.Array
) -> jax.Array:
    """
    compute cov(Y(t),Y(T)|Z_s=z) with s<t<=T


    Args:
        s: time points of shape (batch_size,)
        t: time points of shape (batch_size,)
        T: time-horizon / largest time-step that is considered (float)
        gamma: MAfBM weights of shape (K,)

    Returns:
        cov(Y(t),Y(T)|Z_s=z)
    """
    # shape sanity
    if not (isinstance(T, float) or isinstance(T, int)):
        T = rearrange(squeeze(T), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    s = rearrange(squeeze(s), 'B -> B 1 1')
    gamma_i = rearrange(squeeze(gamma), 'K -> 1 K 1')
    gamma_j = rearrange(squeeze(gamma), 'K -> 1 1 K')
    # compute
    gamma_ij =  gamma_i + gamma_j
    exponentials = jnp.exp(-(T-t) * gamma_j) \
                   - jnp.exp(-(T-s) * gamma_j) * jnp.exp(-(t-s) * gamma_i)
    return exponentials / gamma_ij


def covZ(
    t: jax.Array,
    T: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    s: float = None,
    eps: float = 1e-3
) -> jax.Array:
    """
    compute covariance for all of Z

    Args:
        t: time points of shape (batch_size,)
        T: time-horizon / largest time-step that is considered (float)
        omega: MAfBM weights of shape (K,)
        gamma: MAfBM weights of shape (K,)
        g: value of diffusion function g (?)
        s: time points of shape (batch_size,)
        eps: regularization of matrix (for inversion)

    Returns:
        covariance of Z
    """
    # shape sanity
    t, omega, gamma = map(lambda e: squeeze(e), (t, omega, gamma))
    # fetch number of augmented processes and batch-size
    K = omega.shape[0]
    bs = t.shape[0]
    # initialize empty cov matrix
    s = jnp.zeros(t.shape) if s is None else s
    Sig = jnp.zeros([bs, K+1, K+1])
    # variance of X
    #Sig[:,0,0] = covX(s,t,T, omega, gamma, g)
    Sig_xx = covX(s, t, T, omega, gamma, g, keepdims=True)
    # mixed moments
    Sig_yx = covYX(t, T, omega, gamma, g, keepdims=True)
    Sig_xy = Sig_yx.transpose(0, 2, 1)
    # variance of Y processes
    Sig_yy = covY(s, t, T, gamma)
    # build covariance matrix
    Sig = jnp.block([[Sig_xx, Sig_xy],
                     [Sig_yx, Sig_yy]])
    # regularization
    reg = jnp.eye(K+1, K+1)[None, :, :] * jnp.ones([bs, K+1, K+1]) * eps
    Sig = Sig + reg
    # sanity checking covariance matrix
    pos_var = (jnp.diag(Sig[0]) > 0)
    #assert pos_var.all(), f'Found negativ variance:\n{pos_var}'
    return Sig
