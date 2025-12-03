r"""

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jax import Array


@dataclass
class Logger(ABC):
    log_dir: str

    @abstractmethod
    def log_scalar(
        self, name: str, value: float, log_section: str = 'pretraining'
    ) -> None:
        """
        Simple function for logging of scalar values

        Args:
            name (str): variable name
            name (float): value of scalar variable
        """
        return NotImplemented

    @abstractmethod
    def log_images(
        self, name: str, imgs: Array, log_section: str = 'pretraining'
    ) -> None:
        """
        Simple function for logging of scalar images 

        Args:
            name (str): variable name
            name (Array): value of scalar variable
        """
        r
        return NotImplemented


class PrintLogger(Logger):
    """Dummy Logger that only prints logging information"""

    def log_scalar(
        self, name: str, value: float, log_section: str = 'pretraining'
    ) -> None:
        """See `Logger`."""
        print(name, value, log_section, flush=True)

    def log_images(
        self, name: str, imgs: Array, log_section: str = 'pretraining'
    ) -> None:
        """See `Logger`."""
        print(name, imgs.shape, log_section, flush=True)
