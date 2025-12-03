r"""
This module provides a simple sharding to replicate data parallel in JAX.


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from .jax_typing import PyTree
from typing import List
from jax import Array


def shard_params(
    params: PyTree[Array],
    param_sharding: NamedSharding
) -> PyTree:
    """In data parallel, we replicate the weights on all accelerator devices"""
    return jax.tree_map(lambda x: jax.device_put(x, param_sharding), params)


def map_sharding(sharding: NamedSharding, *args: List[Array]) -> List[Array]:
    """applies the same sharding to all Arrays"""
    return [jax.device_put(arg, sharding) for arg in args]


# number of accelerator devices
DEV_COUNT = jax.device_count()


# name sharding axis
AXIS_NAME = 'batch'


# mesh setup, sharding along the batch axis for data parallel
MESH = jax.make_mesh((DEV_COUNT,), (AXIS_NAME,))


# sharding setup
DP_SHARDING = NamedSharding(MESH, P(AXIS_NAME))
REPLICATE_SHARDING = NamedSharding(MESH, P())
