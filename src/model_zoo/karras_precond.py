r"""
This module provides methods for Karras-like scaling in Schrödinger Bridge 
Flows (see Appendix J of Bortolli et al.).


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax.numpy as jnp


def append_dims(tensor, reference):
    for _ in range(len(reference.shape)-len(tensor.shape)):
        tensor = tensor[..., None]
    return tensor


def c_in(t, eps):
    ci_sq = 1 / (1 + (eps-2) * t * (1-t))
    return jnp.sqrt(ci_sq)


def c_skip(t, eps):
    return ((eps-2) * t - 1 ) / (1 + (eps-2) * t * (1-t))


def c_out(t, eps):
    co_sq = (1 + t + (eps-2) *  t * (1-t)) / (1-t)
    return jnp.sqrt(co_sq)


def predict(t, eps, apply_fn, params, x_t, *args, **kwargs):
    if eps is not None:
        ci = c_in(t, eps)
        co = c_out(t, eps)
        cs = c_skip(t, eps)
        ci, co, cs = map(lambda e: append_dims(e, x_t), (ci, co, cs))
    else:
        ci = 1
        co = 1
        cs = 0
    return co * apply_fn(params, ci*x_t, *args, **kwargs) + cs * x_t
