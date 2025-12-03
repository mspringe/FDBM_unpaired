r"""
Helper functions and a class that handles logging of scalar values and images
via WandB.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import os
import wandb
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from .base import Logger
from ..data import to_npuint8, unscale_vision
from typing import Any


def make_grid(
    imgs: jax.Array,
    num_cols: int,
    padding: int = 2,
    add_alpha_ch: bool = True,
    max_val: float = 1.0
) -> jax.Array:
    """
    Similarly to torchvision's make_grid, this function creates a grid of
    images from jax arrays.

    Args:
        imgs (jax.Array): array of images with shape (N, H, W, C)
        num_cols (int): number of columns in grid
        padding (int): minimum padding between images
        add_alpha_ch (bool): if True, an alpha-channel will be added
        max_val (float): of add_alpha_ch is set, the max. value needstp ne
            known for alpha-channels (e.g. 1.0 or 255)

    Return:
        (jax.Array) an image with all images stored as a grid of shape (H, W, C)
    """
    # establishing the canvas size
    N, H, W, C = imgs.shape
    C = 3 if (C == 1 and add_alpha_ch) else C
    n_cols = min(num_cols, N)
    n_rows = N // n_cols + int((N / n_cols) % 1.0  > 0)
    # creating an empty canvas to draw/copy images on
    grid = jnp.zeros(
        (padding + n_rows * (H+padding),
         padding + n_cols * (W+padding),
         C + int(add_alpha_ch)),
        jnp.float32
    )
    # filling the canvas with the grid of images
    row = 0
    for i, img in enumerate(imgs):
        col = i % n_cols
        # image content
        grid = grid.at[padding+(H+padding)*row:(H+padding)*(row+1),
                       padding+(W+padding)*col:(W+padding)*(col+1),
                       :C].set(img)
        # alpha value (transparent padding-gaps)
        if add_alpha_ch:
            grid = grid.at[padding+(H+padding)*row:(H+padding)*(row+1),
                           padding+(W+padding)*col:(W+padding)*(col+1),
                           -1].set(max_val)
        # simply switching rows every n_cols
        if (i+1) % n_cols == 0:
            row += 1
    return grid


def init_wandb(
    name: str = None,
    key: str = None,
    entity: str = None,
    project: str = 'FDBM',
    group: str = None,
    dir: str = './'
) -> Any:
    """
    This method handles the initialization of a new WandB run, including login
    via your WandB key, if no key is provided or cached, you'll be prompted to
    login via your shell.

    Args:
        name (str): the name of your WandB run
        key (str): your WandB API-key
        entity (str): your WandB username, company-entity, or institute-entity
        project (str): your WandB project
        group (str): grouping of this WandB run withing the project
        dir (str): the path to where WandB logs will be written to.

    Returns:
        the WandB run object
    """
    # they key may either be a string containing the key-code or a path to a 
    # single-line file containing the key-code
    if os.path.isfile(key):
        with open(key, 'r') as f:
            key = f.readline().strip()
    wandb.login(key=key)
    run = wandb.init(project=project, entity=entity, name=name, group=group,
                     dir=dir)
    return run


class WandBLogger(Logger):
    """A simple WandB Logger in compliance with the abstract Logger class"""

    def __init__(
        self,
        log_dir: str,
        wandb_key: str,
        entity: str = None,
        name: str = None,
        project: str = 'FDBM',
        group: str = None,
    ) -> None:
        """
        See `init_wandb`.

        Args:
            log_dir (str): the path to where WandB logs will be written to.
            wandb_key (str): your WandB API-key
            entity (str): your WandB username, company-entity, or institute-entity
            name (str): the name of your WandB run
            project (str): your WandB project
            group (str): grouping of this WandB run withing the project
        """
        super().__init__(log_dir)
        init_wandb(name=name, key=wandb_key, entity=entity, group=group,
                   dir=log_dir, project=project)

    def log_scalar(
        self, name: str, value: float, log_section: str = 'pretraining',
        step: int = None
    ) -> None:
        """See `log_lib.base.Logger`."""
        try:
            wandb.log({log_section: {name: value}}, step=step, commit=True)
        except Exception as e:
            print(f'{e}\ncould not log {name}, but script will continue',
                  flush=True)

    def log_images(
        self, name: str, imgs: jax.Array, log_section: str = 'pretraining',
        step: int = None
    ) -> None:
        """See `log_lib.base.Logger` and `make_grid`."""
        try:
            # scale from [-1, 1] to [0, 255]
            imgs = unscale_vision(imgs, max_val_out=255., min_val_out=0.)
            # combine images into one big grid image
            grid = make_grid(imgs, 12, padding=4, add_alpha_ch=True,
                             max_val=255)
            # clean conversion to numpy uint8
            grid = to_npuint8(grid, max_val_in=255, min_val_in=0)
            # convert to wandb Image & log
            wandb_img = wandb.Image(grid, caption=name)
            wandb.log({log_section: {f'samples {name}': wandb_img}},
                      step=step, commit=True)
        except Exception as e:
            print(f'{e}\ncould not log {name}, but script will continue',
                  flush=True)
