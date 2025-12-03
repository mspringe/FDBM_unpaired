"""
This module provides simple datasets of common vision dataset. All datasets
inherit from `jax_lib.data.datasets.Dataset`.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import os
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST as TorchMNIST
from torchvision.datasets import EMNIST as TorchEMNIST
import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial
from .dataset import ScaledDataset
from .conversion import from_npuint8
from .latent_datasets import LatentDataset


MNIST_STD = 0.6343224


def load_imgs(root, shape, tensor_fn=jnp.array,
              suffixes=('.jpg', '.jpeg', '.png')):
    """
    Searches all elements in the root directory and sub-directories, loads all
    images with one of the specified suffixes and resizes them.

    Args:
        root: roo directory of images
        shape: image-shape stored as tuple (width, height)
        tensor_fn: function, that converts a numpy array to a tensor
        suffixes: only images with these specified suffixes will be loaded
    Returns:
        list of image-tensors
    """
    assert len(shape) == 2, 'shape should contain width & height'
    imgs = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if any([f.lower().endswith(sfx) for sfx in suffixes]):
                image = Image.open(os.path.join(r, f))
                resized_image = image.resize(shape, Image.LANCZOS)
                imgs.append(tensor_fn(np.array(resized_image)))
    return imgs


class MNISTLike(ScaledDataset):
    """
    This class builds a `jax_lib.data.dataset.Dataset` from a torchvision
    dataset with variable conventions like `torchvision.datasets.MNIST`.
    """

    def __init__(self, constructor, root, train=True, dtype=jnp.float32,
                 transpose_X=False, cls_filter=None, verbose=False,
                 scale=MNIST_STD, **kwargs):
        dset = constructor(root, train=train, download=True, **kwargs)
        # torch tensor -> numpy & channel last convention
        X = dset.data.squeeze()[:,:,:,None]  # values in [0, 255]
        X = X.numpy()
        # Note: EMNIST has to get transposed
        if transpose_X:
            X = X.transpose(0, 2, 1, 3)
        # numpy -> jax.numpy
        X = from_npuint8(X, max_val_out=1, min_val_out=-1, dtype=dtype)
        # enforce flattened array
        Y = dset.targets.squeeze()
        # torch tensor -> jax.numpy
        Y = jnp.array(Y.numpy(), dtype=jnp.int32)
        # filter classes (e.g. letters a-e & A-E for EMNIST)
        if cls_filter is not None:
            fltr = jnp.isin(Y, cls_filter)
            Y = Y[fltr]
            X = X[fltr]
        super().__init__(X=X, Y=Y, verbose=verbose, mean=0, std=scale)


class MNIST(MNISTLike):

    def __init__(self, root, train=True, dtype=jnp.float32, verbose=True,
                 scale=MNIST_STD, **kwargs):
        super().__init__(TorchMNIST, root, train=train, dtype=dtype,
                         verbose=verbose, scale=scale)


class EMNIST(MNISTLike):
    """
    We closely follow the setup of
        Shi et al. (2023)
        De Bortoli et al. (2021)
    and train the models to transfer between 10 EMNIST letters, A-E and a-e.
    """

    def __init__(self, root, train=True, dtype=jnp.float32, verbose=True,
                 scale=MNIST_STD, **kwargs):
        super().__init__(TorchEMNIST, root, train=train, dtype=dtype,
                         split='letters', transpose_X=True,
                         cls_filter=jnp.arange(5), verbose=verbose,
                         scale=scale)


class AFHQ(ScaledDataset):
    """
    This class loads a subsets of the 3 animal classes from the AFHQ dataset
    """

    def __init__(self, root, animal='cat', size=64, train=True, verbose=True,
                 dtype=jnp.float32):
        # image sub-directory & shape
        subdir = os.path.join(root, 'AFHQ', 'train' if train else 'val', animal)
        shape = (size, size)
        # converts images to jax tensors in [-1, 1] of specified data-type
        tensor_fn = partial(from_npuint8, max_val_out=1, min_val_out=-1,
                            dtype=dtype)
        # select moments
        mean = 0
        if size == 32:
            std = 0.4641
        elif size == 64:
            std = 0.4778
        else:
            std = None
        # load images
        X = jnp.stack(load_imgs(subdir, shape=shape, tensor_fn=tensor_fn))
        # store dummy classes for duck-typing
        Y = jnp.zeros(len(X))
        super().__init__(X=X, Y=Y, verbose=verbose, mean=mean, std=std)


class AFHQ_Cat(AFHQ):
    """This class loads cats subset of the AFHQ dataset"""

    def __init__(self, root, size=64, train=True, dtype=jnp.float32,
                 verbose=True):
        super().__init__(root, 'cat', size=size, train=train, dtype=dtype,
                         verbose=verbose)


class AFHQ_Wild(AFHQ):
    """This class loads wild subset of the AFHQ dataset"""

    def __init__(self, root, size=64, train=True, dtype=jnp.float32,
                 verbose=True):
        super().__init__(root, 'wild', size=size, train=train, dtype=dtype,
                         verbose=verbose)


class AFHQ_Dog(AFHQ):
    """This class loads dogs subset of the AFHQ dataset"""

    def __init__(self, root, size=64, train=True, dtype=jnp.float32,
                 verbose=True):
        super().__init__(root, 'dog', size=size, train=train, dtype=dtype,
                         verbose=verbose)
