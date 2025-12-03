"""
This module provides SDE & ODE solver methods for SB-models.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
import jax.numpy as jnp
import math
from functools import partial
from ..model_zoo import predict
from .. import distributed as dist


@partial(jax.jit, static_argnames=['apply_fn', 'train'])
def eval_model(apply_fn, params, x, *args, **kwargs):
    return apply_fn(params, x, *args, **kwargs)


def em(apply_fn, params, x, directions, num_steps=40, t_offset=0, eps=1,
       rngs=None, verbose=False, key=None, karras_pred=False):
    """
    Euler-Maruyama SDE solving method

    Args:
        apply_fn: apply function a flax model
        params: parameterization of model
        directions: direction of SB (either 1s or 0s)
        num_steps: number of discretization steps for solver
        t_offset: offsets the time interval to [t_offset, 1-t_offset]
        rngs: (optional) random number generators, if e.g. dropout should be
              used during inference, rather than averaged
        key: PRNG key for stochasticity during sampling
    Returns:
        samples of the Euler-Maruyma method
    """
    # sharding
    x = jax.device_put(x, dist.DP_SHARDING)
    params = jax.device_put(params, dist.REPLICATE_SHARDING)
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
    dt = 1 / num_steps  #jnp.diff(ts).mean()  # accounting for float-rounding errors via mean
    # discretized rate of change w.r.t. brownian motion
    dw = math.sqrt(dt)  #jnp.sqrt(dt)
    # scaling / variance arising from entropic regularization
    eps_scale = math.sqrt(eps)
    # setting the RNG key, if not specified
    key = jax.random.PRNGKey(42) if key is None else key
    # if `verbose`, we will print a progress bar
    iterator = tqdm(ts) if verbose else ts
    # Euler-Maruyma solver steps
    # apply initial noise
    key, subkey = jax.random.split(key)
    #x = x + math.sqrt(eps * (1-t_offset) * t_offset) \
    #        * jax.random.normal(subkey, shape=x.shape)
    for c, i in enumerate(iterator):
        # updating RNG keys
        key, subkey = jax.random.split(key)
        # same point in time for all batch-elements
        t = _1 * i
        # modelling the drift function f via neural network
        args = dist.map_sharding(dist.DP_SHARDING, t, directions)
        kwargs = dict(rngs=rngs, train=train)
        if karras_pred:
            f = predict(t, eps, apply_fn, params, x, *args, **kwargs)
        else:
            f = eval_model(apply_fn, params, x, *args, **kwargs) #apply_fn(params, x, *args, **kwargs)
        # modelling the diffusion function g via sampling and entropic reg.
        if (c+1) < len(ts):
            g = eps_scale * jax.random.normal(subkey, shape=x.shape)
        else:
            g = 0
        # calculating the discretized rate of change w.r.t. x
        dx = f * dt + g * dw
        # discretized update of x
        x = x + dx
    return x


def pf(apply_fn, params, x, directions, num_steps=20, t_offset=0, eps=1,
       rngs=None, verbose=False, karras_pred=False, key=None):
    """
    Euler (PF) ODE solving method following Bortoli et al. 2024
    (see arXiv:2409.09347).

    Args:
        apply_fn: apply function a flax model
        params: parameterization of model
        directions: direction of SB (either 1s or 0s)
        num_steps: number of discretization steps for solver
        t_offset: offsets the time interval to [t_offset, 1-t_offset]
        rngs: (optional) random number generators, if e.g. dropout should be
              used during inference, rather than averaged
    """
    train = rngs is not None
    ts = jnp.linspace(0+t_offset, 1-t_offset, num_steps, dtype=x.dtype)
    dt = 1/num_steps
    N = x.shape[0]
    iterator = tqdm(ts) if verbose else ts
    for i in iterator:
        t = jnp.ones((N,), dtype=x.dtype) * i
        args_1 = (t, directions)
        args_0 = (1-t, 1-directions)
        kwargs = dict(rngs=rngs, train=train)
        if karras_pred:
            drift_1 = predict(t, eps, apply_fn, params, x, *args_0, **kwargs)
            drift_0 = predict(1-t, eps, apply_fn, params, x, *args_1, **kwargs)
        else:
            drift_1 = eval_model(apply_fn, params, x, *args, **kwargs) #apply_fn(params, x, *args_0, **kwargs)
            drift_0 = eval_model(apply_fn, params, x, *args, **kwargs) #apply_fn(params, x, *args_1, **kwargs)
        dx = 0.5 * (drift_1 - drift_0) * dt
        x = x + dx
    return x
