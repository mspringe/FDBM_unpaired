"""
This module provides some additional typing variables for JAX, such as PyTrees.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from typing import TypeVar, Union, Mapping, Sequence, Generic, Callable


T = TypeVar("T")
PyTree = Union[T, Sequence["PyTree[T]"], Mapping[str, "PyTree[T]"]]
LossScale = Callable
