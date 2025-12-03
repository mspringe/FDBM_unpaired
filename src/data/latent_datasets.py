"""
This module provides simple datasets of precomputed latent representations of
your data.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import os
import numpy as np
import torch
import jax.numpy as jnp
from diffusers.models import AutoencoderKL
from .dataset import ScaledDataset


def load_latents(pth, tensor_fn=jnp.array):
    return tensor_fn(np.load(pth).astype(np.float32))


class LatentDataset(ScaledDataset):

    def __init__(
        self,
        pth: str,
        vae: AutoencoderKL, 
        verbose: bool = False
    ) -> None:
        X = load_latents(pth)
        # channel last convention
        X = X.transpose(0, 2, 3, 1)
        Y = jnp.zeros(len(X))
        self.vae = vae
        super().__init__(X=X, Y=Y, verbose=verbose,
                         mean=0, std=1/self.vae.config.scaling_factor)

    def decode(self, latents, scale_pm1=True, device='cpu'):
        # rescale to initial latent space scale
        latents = self.unscale(latents)
        # channel first convention
        latents = latents.transpose(0, 3, 1, 2)
        # convert to torch tensor
        latents = torch.from_numpy(np.array(latents).astype(np.float32))
        # decode to pixel-space
        with torch.no_grad():
            self.vae = self.vae.eval().to(device)
            preds = self.vae.decode(latents.to(device)).sample
            preds = jnp.array(preds.cpu().numpy())
            # channel last convention
            preds = preds.transpose(0, 2, 3, 1)
        # if specified, scale to [-1, 1]
        if scale_pm1:
            preds = jnp.clip(preds, 0, 1)
            preds = (preds * 2) - 1
        return preds


class Latent_AFHQ_Cat_256(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_256_cat_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_256_cat_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)


class Latent_AFHQ_Wild_256(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_256_wild_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_256_wild_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)


class Latent_AFHQ_Cat_512(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_512_cat_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_512_cat_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)


class Latent_AFHQ_Wild_512(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_512_wild_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_512_wild_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)


class Latent_AFHQ_Dog_256(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_256_dog_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_256_dog_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)


class Latent_AFHQ_Dog_512(LatentDataset):

    def __init__(self, root='datasets', verbose=True, train=True, **kwargs):
        if train:
            fname = os.path.join('latents', 'sd-vae-ft-ema_512_dog_lat.npy')
        else:
            fname = os.path.join('latents', 'sd-vae-ft-ema_val_512_dog_lat.npy')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        super().__init__(os.path.join(root, fname), vae, verbose=verbose)
