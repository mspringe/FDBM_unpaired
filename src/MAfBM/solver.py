"""
This module provides the SDE solver methods.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from einops import rearrange
import jax
import jax.numpy as jnp
import math
from functools import partial
from ..model_zoo import predict
from .fsb import input_transform, score_u, fSB
from .util import matrix_vector_mp
from typing import Callable, Dict, List
from ..jax_typing import PyTree


@partial(jax.jit, static_argnames=['apply_fn', 'train'])
def eval_model(
    apply_fn: Callable,
    params: PyTree[jax.Array],
    x: jax.Array,
    *args: List[jax.Array],
    **kwargs: Dict[str, jax.Array]
) -> jax.Array:
    """jit compiled neural network call"""
    return apply_fn(params, x, *args, **kwargs)


def em_fwd(
    apply_fn: Callable,
    params: PyTree[jax.Array],
    fsb: fSB,
    x: jax.Array,
    directions: jax.Array,
    num_steps: int = 40,
    t_offset: float = 0,
    Y_0: jax.Array = 0,
    rngs: PyTree[jax.random.PRNGKey] = None,
    verbose: bool = False,
    key: jax.random.PRNGKey = None
) -> jax.Array:
    """
    Euler-Maruyama SDE solving method

    Args:
        apply_fn (Callable): apply function a flax model
        params (PyTree): parameterization of model
        directions (jax.Array): direction of SB (either 1s or 0s)
        num_steps (int): number of discretization steps for solver
        t_offset (float): offsets the time interval to [t_offset, 1-t_offset]
        Y_0 (jax.Array): initialization of augmenting processes, degaults to 0
        rngs (PyTree): random number generators, if e.g. dropout should be
            used during inference, rather than averaged
        verbose (bool): if this flag is set, a tqdm progress bar will be 
            displayed
        key: PRNG key for reproducable stochasticity during sampling

    Returns:
        samples from the Euler-Maruyma method
    """
    # if `train` is false, this method averages over weights for dropout
    train = rngs is not None
    # shape sanity
    directions = directions.squeeze()
    # get batch size
    N = x.shape[0]
    # tensor of 1s to quickly map a scalar to a JAX tensor
    _1 = jnp.ones((N,), dtype=x.dtype)
    # equidistant points in [0+offset, 1-offset]
    ts = jnp.linspace(0+t_offset, 1-t_offset, num_steps, dtype=x.dtype)
    # discretized rate of change w.r.t. time
    dt = 1 / num_steps
    # setting the RNG key, if not specified
    key = jax.random.PRNGKey(42) if key is None else key
    # if `verbose`, we will print a progress bar
    iterator = tqdm(ts) if verbose else ts
    # Euler-Maruyma solver steps
    y = jnp.ones([*x.shape,fsb.K]) * Y_0
    z = jnp.concatenate([x[...,None],y],axis=-1)
    T = _1
    G = rearrange(fsb.G, 'M -> 1 1 1 1 M' )
    GG = G[...,None] @ G[...,None,:]
    F = rearrange(fsb.F, 'M N -> 1 1 1 1 M N')
    for c, i in enumerate(iterator):
        # updating RNG keys
        key, subkey = jax.random.split(key)
        # same point in time for all batch-elements
        t = _1 * i
        # modelling the drift function f via neural network
        args = (t, directions)
        kwargs = dict(rngs=rngs, train=train)
        x = input_transform(
            z[...,0], z[...,1:], t, T, fsb.omega, fsb.gamma, fsb.g_max,
            direction=1
        )
        score_x = eval_model(apply_fn, params, x, *args, **kwargs)
        u = score_u(score_x, t, T, fsb.omega, fsb.gamma, fsb.g_max)
        # modelling the diffusion function g via sampling and entropic reg.
        if (c+1) < len(ts):
            # (B,W,H,C,1)
            noise = jax.random.normal(subkey, shape=(*z[...,0].shape,1))
            dw = jnp.sqrt(dt) * noise
        else:
            dw = 0
        dz = (matrix_vector_mp(F, z) + matrix_vector_mp(GG, u))*dt + G * dw
        # discretized update of z
        z = z + dz
    return z
