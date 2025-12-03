"""
This module provides high-level functionalities to apply MAfBM to
Schrödinger Bridge Flow.

Exemplary use-case:

>>> # assume some inputs and parameters
>>> key = jax.random.PRNGKey(0)
>>> T = 1.0
>>> eps = 1.0
>>> g = math.sqrt(eps)
>>> t = jax.numpy.linspace(key, 0, T, 128)
>>> x0 = jax.numpy.zeros((128, 32, 32, 3))
>>> xT = jax.numpy.ones((128, 32, 32, 3))
>>> # sampling from marginals of fSB
>>> fsb = fSB(H=0.9, K=5)
>>> key, subkey = jax.random.split(key)
>>> y0 = init_Y_terminal(x0, fsb.omega, fsb.gamma, fsb.g_max)
>>> zt = sample_pinned(subkey, t, T, x0, xT, fsb.omega, fsb.gamma, g, Y_0=y0)
>>> xt = input_transform(zt[..., 0], zt[..., 1:], t, T, fsb.omega, fsb.gamma, g)

All code was written for a markovian approximation of type II Brownian motion.


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import math
import jax
from jax.lax import select, cond
import jax.numpy as jnp
import jax.scipy as jscp
from einops import rearrange
from functools import partial
from . import mafbm
from .mixed_moments import covX, covY, covZ, covYX, zeta
from .util import squeeze, matrix_vector_mp


class fSB:
    """A simple helper class to wrap all neccessary variables for fSB."""

    def __init__(
        self,
        H=0.5,
        K=5,
        T=1.0,
        norm=True,
        g_max=1.0,
        gamma_max=20.0,
        gamma_min=0.1,
        fbm_type=2
    ) -> None:
        # storing hyper-parameters
        self.H = H
        self.K = K
        self.T = T
        self.norm = norm
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.g_max = g_max
        self.fbm_type = fbm_type
        # calculating weights
        omega, gamma = mafbm.calc_omega_gamma(H, K, gamma_max, gamma_min, T)
        # optional normalization of omega
        if self.norm and K>1:
            assert T==1, f'cannot normalize for T={T}'
            _1 = jnp.ones([1])
            var_T = cond_var(_1*0, _1*T, omega, gamma, 1.0)
            blue = '\033[94m'
            neutral = '\033[0m'
            print(f'{blue}[MAfBM]{neutral} normalizing MAfBM with',
                  float(jnp.sqrt(var_T).squeeze()))
            omega = omega / jnp.sqrt(var_T)
        # storing calculated variables
        self.gamma = gamma
        self.omega = omega
        if self.fbm_type == 2:
            self.F = - jnp.concatenate(
                [jnp.zeros([K+1,1]),
                 jnp.vstack([self.g_max*(self.omega * self.gamma),
                             jnp.diag(self.gamma)])],
                axis=-1
            )
            self.G = jnp.array([jnp.sum(self.omega) * self.g_max,
                                *jnp.ones(self.K)])
        else:
            raise NotImplementedError(f'{self.fbm_type} not implemented yet.'
                                      f' Only Type 2 available.')

    def __str__(self) -> str:
        """A tabular representation of relevant variables."""
        blue = '\033[94m'
        neutral = '\033[0m'
        cstrip = lambda s: s.replace(blue, '').replace(neutral, '')

        def frame(s, head_lines=2):
            lines = s.split('\n')
            max_space = max(map(len, [cstrip(l) for l in lines]))
            mod_lines = []
            # top rule
            mod_lines.append(f'┌{"─" * max_space}┐')
            # header
            for line in lines[:head_lines]:
                mod_lines.append(f'│{line}{" "*(max_space-len(cstrip(line)))}│')
            # mid rule
            mod_lines.append(f'├{"-" * max_space}┤')
            # content
            for line in lines[head_lines:]:
                mod_lines.append(f'│{line}{" "*(max_space-len(cstrip(line)))}│')
            # bottom rule
            mod_lines.append(f'└{"─" * max_space}┘')
            return '\n'.join(mod_lines)

        def tabulate(name, contents, space, multiline):
            if not multiline:
                return f'\n{blue}{name}{neutral}' \
                       f"{' ' * (space-len(name))}{contents}"
            out = f'\n{blue}{name}{neutral}' \
                  f"{' ' * (space-len(name))}{contents[0]}"
            for line in contents[1:]:
                out += f"\n{' ' * space}{line}"
            return out

        # all variables
        variables = vars(self)
        # headers
        h1, h2 = f'{blue}Variable{neutral}', f'{blue}Value{neutral}'
        # min spacing
        space = 1 + max([len(k) for k in variables.keys()] +
                        [len(cstrip(h1))])
        # build table
        vars_table = ''
        for k, v in variables.items():
            multiline = False
            if isinstance(v, jnp.ndarray):
                # rounded scalars in vector
                if len(v.shape) == 1:
                    v = list(map(partial(round, ndigits=2), v.tolist()))
                # rounded scalars in matrix
                elif len(v.shape) == 2:
                    multiline = True
                    v = [list(map(partial(round, ndigits=2), e.tolist()))
                         for e in v]
                # attempting rounding of scalar
                else:
                    v = round(float(v), 2)
            vars_table += tabulate(k, v, space, multiline)
        vars_table = f'{blue}fSB{neutral}\n'\
                     f"{h1}{' ' * (space-len(cstrip(h1)))}{h2}" + vars_table
        # frame table
        vars_table = frame(vars_table)
        return vars_table


def fsb_from_args(args) -> fSB:
    """Constructs all MAfBM variable from command line args"""
    return fSB(H=args.H, K=args.K, norm=args.norm) if H is not None else None


def Y_0_sanity(Y_0: jax.Array) -> jax.Array:
    """
    This method flattens all feature dimensions of Y_0.

    Args:
        Y_0 (jax.Array): tensor of shape (B,D,K), (B,L,D,K), or (B,H,W,C,K)
    Returns:
        (jax.Array) Y_0 of shape (B, (...), K)
    """
    if not isinstance(Y_0, int) and not isinstance(Y_0, float):
        num_axes = len(Y_0.shape)
        if num_axes == 4:
            Y_0 = rearrange(Y_0, 'B L D K -> B (L D) K')
        elif num_axes == 5:
            Y_0 = rearrange(Y_0, 'B H W C K -> B (H W C) K')
        elif num_axes != 3:
            raise ValueError('Y_0 shape should be (B,D,K), (B,L,D,K), or '
                             '(B,H,W,C,K)')
    return Y_0


def sample_batch_mvn(
    key: jax.random.PRNGKey,
    cov_matrix: jax.Array,
    c: int = 2,
    h: int = 1,
    w: int = 1,
    batch_size: int = 128,
    aug_dim: int = 6,
    channel_last: bool = True
) -> jax.Array:
    """
    Samples i.i.d. from a multivariate normal distribution with specified
    covariance and zero-mean.

    Args:
        key (jax.random.PRNG): RNG key for reproducibility
        cov_matrix (jax.Array): batch with shape (B, K+1, K+1)
        c (int): dimensionality of channels for output
        h (int): dimensionality of height for output
        w (int): dimensionality of width for output
        batch_size (int): batch_size B
        aug_dim (int): number of augmented processes K+1
        channel_last (bool): flag indicating whether to use channel-last
            convention for images

    Returns:
        (jax.Array) samples from N(0, I)
    """
    # zero-mean
    mean = jnp.zeros([batch_size, 1, aug_dim])
    # Sample from the distribution
    D = int(c*h*w)
    shape = (batch_size, D)
    cov_matrix = rearrange(cov_matrix, 'B M N -> B 1 M N')
    samples = jax.random.multivariate_normal(key, mean, cov_matrix, shape)
    # reshape to respective tensor convention
    if channel_last:
        format_str = 'B (H W C) K -> B H W C K'
    else:
        format_str = 'B (C H W) K -> B C H W K'
    samples = rearrange(samples, format_str, C=c, H=h, W=w)
    return samples


@partial(jax.jit, static_argnames=('channel_last', 'Y_0', 'g'))
def sample_pinned(
    key: jax.random.PRNGKey,
    t: jax.Array,
    T: float,
    x0: jax.Array,
    xT: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    Y_0: float = 0,
    channel_last: bool = True
) -> jax.Array:
    """
    The reciprocal interpolation between z_0 and z_t, given x0 and xT at point
    in time t for fSB.

    Args:
        key (jax.random.PRNGKey): RNG key for reproducability
        t (jax.Array): points in time of shape (batch_size,)
        T (float): maximium point in time, e.g. 1.0
        x0 (jax.Array): samples of pi_0 with shape (batch_size, dim),
            where dim = C*H*W
        xT (jax.Array): samples of pi_T with shape (batch_size, dim),
            where dim = C*H*W
        omega (jax.Array): MAfBM weights of shape (K,)
        gamma (jax.Array): MAfBM geometric grid of shape (K,)
        g (float): value of diffusion function g at t 
            (e.g. constant std of entropic reg.)
        channel_last: if true, channels are stored last, otherwise after batch

    Returns:
        (jax.Array) samples z_t from the pinned process at point in time t
    """
    Y_0 = Y_0_sanity(Y_0)
    # shape sanity
    t = rearrange(squeeze(t), 'B -> B 1 1')
    s = jnp.zeros_like(t)
    omega = rearrange(squeeze(omega), 'K -> 1 K 1')
    gamma = rearrange(squeeze(gamma), 'K -> 1 K 1')
    # retrieve dimensionality
    K = omega.shape[1]
    if channel_last:
        bs, h, w, c = x0.shape
        shape_str = 'H W C'
    else:
        bs, c, h, w = x0.shape
        shape_str = 'C H W'
    d = c*h*w
    # flattened / vectorized
    flat_x0 = rearrange(x0, f'B {shape_str} -> B ({shape_str}) 1')
    flat_xT = rearrange(xT, f'B {shape_str} -> B ({shape_str}) 1')
    # setup mean & cov
    Sig_zx = jnp.concatenate(
        [
            covX(s, t, T, omega, gamma, g, keepdims=True), # (B, 1, 1)
            covYX(t, T, omega, gamma, g, keepdims=True)    # (B, K, 1)
        ],
        axis=1
    )
    Sig_zx = rearrange(Sig_zx, 'B M 1 -> B M')             # (B, K+1)
    prod_zx = Sig_zx[:,:,None] * Sig_zx[:,None,:]          # (B, K+1, K+1)
    var = covX(s, T, T, omega, gamma, g, keepdims=True)    # (B, 1, 1)
    Sig_bar = covZ(t,t,omega,gamma,g) - (1/var) * prod_zx  # (B, K+1, K+1)
    flat_mu = jnp.concatenate(  
        [
            flat_x0,
            jnp.ones([bs, d, K]) * Y_0
        ],
        axis=-1
    )
    flat_diff = flat_xT - flat_x0
    flat_mu_bar = flat_mu + (1/var) * Sig_zx[:,None,:] * flat_diff # (B, D, K+1)
    # sample
    noise = sample_batch_mvn(key, Sig_bar, c=c, h=h, w=w, batch_size=bs,
                             aug_dim=K+1, channel_last=channel_last)
    # unravel dimensions C, H, and W
    mu_bar = rearrange(flat_mu_bar, f'B ({shape_str}) K -> B {shape_str} K',
                       C=c, H=h, W=w)
    # compute new z_t
    zt = mu_bar + noise
    return zt


@jax.jit
def input_transform(
    x: jax.Array,
    Y: jax.Array,
    t: jax.Array,
    T: float,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    direction: int = 1
) -> jax.Array:
    """
    This method calculates the conditional mean E[X_1 | Z_t=z_t]
    The conditional mean is a nice trick to embed all necessary information 
    contained in Z w.r.t. X into a tensor representation that has the same 
    dimensionality as X, regardless of K.

    Args:
        x (jax.Array): current data sample at time t
        Y (jax.Array): samples of Y-processes at time t
        t (jax.array): point in time from interval [0, T]
        T (float): maximum point in time
        omega (jax.Array):  MAfBM weights of shape (K,)
        gamam (jax.Array):  MAfBM geometric grid of shape (K,)
        g (float): value of diffusion function g at t 
            (e.g. constant std of entropic reg.)
        direction (int): directions of shape (K,), 
            where 1 -> forward and 0 -> backward

    Returns:
        (jax.Array) a conditional mean representation of X w.r.t. Z
    """
    # shape sanity
    if not (isinstance(T, float) or isinstance(T, int)):
        T = rearrange(squeeze(T), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    gamma = rearrange(squeeze(gamma), 'K -> 1 1 K')
    omega = rearrange(squeeze(omega), 'K -> 1 1 K')
    # compute input transform
    weight = omega * zeta(t, T, gamma, g)
    # match dim if Y has more than one feature dim (e.g. also H W C)
    while len(weight.shape) < len(Y.shape):
        weight = weight[...,None,:]
    y_part = jnp.sum(weight*Y, axis=-1)
    sign = (2*direction-1)  # addition for fwd, substraction for bwd
    return x + sign * y_part


def score_u(
    score_x: jax.Array,
    t: jax.Array,
    T: float,
    omega: jax.Array,
    gamma: jax.Array,
    g: jax.Array,
    # y: jax.Array = None
) -> jax.Array:
    """
    Takes the he learned drift for X and calculate the drift for Z

    Args:
        score_x (jax.Array): score w.r.t. x (learned via neural network)
        t (jax.Array): point in time
        T (float): maximum of the time interval
        omega (jax.Array): MAfBM weights
        gamma (jax.Array): MAfBM geometric grid
        g (float): value of diffusion function g at t 
            (e.g. constant std of entropic reg.)

    Returns:
        (jax.Array) learned drift for Z
    """
    # shape sanity
    if not (isinstance(T, float) or isinstance(T, int)):
        T = rearrange(squeeze(T), 'B -> B 1 1')
    t = rearrange(squeeze(t), 'B -> B 1 1')
    omega = rearrange(squeeze(omega), 'K -> 1 1 K')
    gamma = rearrange(squeeze(gamma), 'K -> 1 1 K')
    # scaling and unravelling of Z
    zta = zeta(t, T, gamma, g)           # shape B 1 K
    _1 = jnp.ones([*zta.shape[:-1], 1])  # shape B 1 1
    scale = jnp.concatenate(
        [
            _1,          # X
            omega * zta  # Y_1, ..., Y_K
        ],
        axis=-1
    )  # shape B 1 (K+1)
    # match dims
    score_x = score_x[..., None]
    while len(scale.shape) != len(score_x.shape):
        scale = scale[...,None,:]
    return scale * score_x


def cond_var(
    s: jax.Array,
    t: jax.Array,
    omega: jax.Array,
    gamma: jax.Array,
    g: float,
    keepdims: bool = False
) -> jax.Array:
    """
    compute cov(X(t), X(t)|Z_s=z) with s<t

    Args:
        t (jax.Array): time points of shape (batch_size,)
        s (jax.Array): time points of shape (batch_size,)
        T (float): time-horizon / largest time-step that is considered (float)
        omega (jax.Array): MAfBM weights of shape (K,)
        gamma (jax.Array): MAfBM weights of shape (K,)
        g (float): value of diffusion function g at t 
            (e.g. constant std of entropic reg.)
        keepdims (bool): flag indicating whether output has shape (B, 1, 1) or (B,)

    Returns:
        (jax.Array) The covariance of X, that is cov(X(t), X(t)|Z_s=z)
    """
    return covX(s, t, t, omega, gamma, g, keepdims=keepdims)
