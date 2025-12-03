"""
This module provides a simple Dataset that you can sample from. Note that data
is assumed to be provided as a jax array and alredy be loaded into RAM.

Note:
    No prefetching of batches is implementing as all data already is located in
    RAM.
    Implement your own custom dataloader, when dealing with I/O.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple
from dataclasses import dataclass
from functools import partial


def idty(x):
    return x


@partial(jax.jit, static_argnames=['num_samples', 'trafo_X', 'trafo_Y'])
def sample(
    key: jax.random.PRNGKey,
    X: jax.Array,
    Y: jax.Array,
    num_samples: int,
    idcs_X: List[int] = None,
    idcs_Y: List[int] = None,
    trafo_X: Callable = idty,
    trafo_Y: Callable = idty
) -> Tuple[jax.Array, jax.Array, jax.random.PRNGKey, jax.random.PRNGKey]:
    """
    Args:
        key (jax.random.PRNGKey): RNG for reproducible data sampling
        X (jax.Array): data of distribution Pi_0
        Y (jax.Array): data of distribution Pi_1
        num_samples (int): batch size
        idcs_X (List[int]): as an alternative to sampling uniformly, you can
            also specify the exact indices of X
        idcs_Y (List[int]): as an alternative to sampling uniformly, you can
            also specify the exact indices of Y
        trafo_X (Callable): augmentations of X
        trafo_Y (Callable): augmentations of Y

    Returns:
        (jax.Array, jax.Array, jax.random.PRNGKey, jax.random.PRNGKey) tuple of
        paring and the RNG keys
    """
    if idcs_X is None:
        key, subkey = jax.random.split(key)
        idcs_X = jax.random.choice(subkey, len(X), shape=(num_samples,),
                                   replace=False)
    if idcs_Y is None:
        key, subkey = jax.random.split(key)
        idcs_Y = jax.random.choice(subkey, len(Y), shape=(num_samples,),
                                   replace=False)
    else:
        key, subkey = key, None
    x = trafo_X(X[idcs_X])
    y = trafo_Y(Y[idcs_Y])
    return x, y, key, subkey


@dataclass
class Dataset:
    X: jnp.ndarray
    Y: jnp.ndarray
    trafo_X: Callable = lambda x: x
    trafo_Y: Callable = lambda x: x
    key = jax.random.PRNGKey(42)

    def __len__(self):
        return len(self.X)

    def sample(self, num_samples, idcs_X=None, idcs_Y=None):
        """
        sampling without duplicates, unless explicitly specified.

        Args:
            num_samples: number of samples to draw
            idcs: (optional) exact indices to draw, overrides num_samples. if
                  None, it will be sampled uniformly without replacement
        Return:
            samples from dataset
        """
        x, y, key, subkey = sample(
            self.key, self.X, self.Y, num_samples, idcs_X=idcs_X, idcs_Y=idcs_Y,
            trafo_X=self.trafo_X, trafo_Y=self.trafo_Y
        )
        self.key = key
        return x, y

    @property
    def shape(self):
        x, y = self.sample(1)
        return x.shape[1:], y.shape[1:]


class ScaledDataset(Dataset):
    """scales and shifts all X-data w.r.t. some mean and std, can be unscaled"""

    def __init__(self, X, Y, verbose=False, mean=None, std=None):
        self.__mean = X.mean() if mean is None else mean
        self.__std = X.std() if std is None else std
        super().__init__(X, Y, trafo_X=self.scale)
        if verbose:
            print(self)

    def scale(self, x):
        return (x-self.__mean) / self.__std

    def unscale(self, x):
        return x * self.__std + self.__mean

    def __str__(self):
        # highlighting blue
        blue = '\033[94m'
        neutral = '\033[0m'
        # table content
        name = f'{blue}ScaledDataset{neutral} {self.__class__.__name__}'
        mean = f'{blue}shift{neutral}  {-1*self.__mean}'
        std = f'{blue}scale{neutral}  {1/self.__std}'
        # padding
        spacing_raw = max(map(len, [name, mean, std]))
        spacing = spacing_raw - len(blue) - len(neutral)
        name_spaces = spacing_raw-len(name)
        mean_spaces = spacing_raw-len(mean)
        std_spaces = spacing_raw-len(std)
        # booktabs
        btab = '─' * spacing
        sep = '-' * spacing
        return f'┌{btab}┐\n' \
               f'│{name}{" "*name_spaces}│\n' \
               f'├{sep}┤\n' \
               f'│{mean}{" "*mean_spaces}│\n' \
               f'│{std}{" "*std_spaces}│\n' \
               f'└{btab}┘'
