"""
This sub-package comprises all functionalities related to the markovian 
approximation of fractional Brownian motion for Schrödinger Bridge Flows.

All code was written for a markovian approximation of type II Brownian motion.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from .mixed_moments import *
from .mafbm import *
from .fsb import *
from .solver import *
from .loss import *
