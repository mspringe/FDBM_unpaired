"""
This module provides a JAX implemention of the DiT architecture; a variant of
the ViT (see arXiv:2010.11929, see also arXiv:2212.09748 for the DiT paper).

All code is heavily based upon the pytorch timm implementation of ViT, see:
github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
github.com/facebookresearch/DiT
github.com/kvfrans/jax-diffusion-transformer


However this implementation features cuDNN flash attention (when possible),
offering some speedup.


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import math
import jax
import jax.numpy as jnp
from jax.random import bernoulli
from jax import lax
import flax.linen as nn
from flax.linen.initializers import xavier_uniform, normal, constant
from einops import rearrange
from .layers import MultiHeadDotProductAttention, LabelEmbedder, \
                              TimestepEmbedder, PatchEmbedder, \
                              qkv_flash_attention, MlpBlock
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union


def scale_shift(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, precision=jnp.float32):
    """
    Args:
        embed_dim: dimension of embedding
        pos: position indices with length L
    Returns:
        sinusoidal embeddings of shape: (L, D)
    """
    assert embed_dim % 2 == 0, 'embedding dim should be an even number'
    # fixed hyper-parameters
    base = 10000
    half_dim = embed_dim // 2
    # embedding
    exponent = - jnp.arange(half_dim, dtype=precision) / half_dim
    omega = base**exponent
    out = jnp.outer(pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Args:
        embed_dim: dimension D of embedding (joint dimensions: height & width)
        grid: grid indices with shape: (height, width)
    Returns:
        jax-numpy array of positional embedding with shape (L, D),
        where L = height * width
    """
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, length, precision=jnp.float32):
    """
    Args:
        embed_dim: embedding dimension D
        length: sequence length L = height * width
        precision: data type, e.g. jax.numpy.float32
    Returns:
        positional embedding of shape: (1, L, D), where L = height * width
    """
    # sanity
    size = math.sqrt(length)
    assert size % 1.0 == 0, "length has to be a cubic number"
    size = int(size)
    # construct pos. embeddings
    grid_h = jnp.arange(size, dtype=precision)
    grid_w = jnp.arange(size, dtype=precision)
    grid = jnp.stack(jnp.meshgrid(grid_w, grid_h), axis=0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed[None,:]
    return pos_embed


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_features: int
    num_heads: int
    mlp_ratio: float = 4.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c, train=True):
        # adaLn modulation parameters
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_features, kernel_init=constant(0.0),
                     dtype=self.dtype, param_dtype=self.param_dtype)(c)
        vals = jnp.split(c, 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals
        # [I.i] Layer Norm (attention)
        x_norm = nn.LayerNorm(
            use_bias=False, use_scale=False, dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        # [I.ii] Scale, Shift (attention)
        x_scale_shiftd = scale_shift(x_norm, shift_msa, scale_msa)
        # [I.iii] MHSA
        attn_x = MultiHeadDotProductAttention(
            kernel_init=xavier_uniform(), num_heads=self.num_heads,
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x_scale_shiftd, x_scale_shiftd)
        # [I.iv] output scale and residual connection
        x = x + (gate_msa[:, None] * attn_x)
        # [II.i] Layer Norm (mlp)
        x_norm2 = nn.LayerNorm(
            use_bias=False, use_scale=False, dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        # [II.ii] Scale, Shift (mlp)
        x_scale_shiftd2 = scale_shift(x_norm2, shift_mlp, scale_mlp)
        # [II.iii] MLP
        mlp_dim = int(self.hidden_features * self.mlp_ratio)
        mlp_x = MlpBlock(
            hidden_features=mlp_dim, dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x_scale_shiftd2, train=train)
        # [II.iv] output scale and residual connection
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    patch_size: int
    out_channels: int
    hidden_features: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(
            2 * self.hidden_features, kernel_init=constant(0),
            dtype=self.dtype, param_dtype=self.param_dtype
        )(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = scale_shift(
            nn.LayerNorm(use_bias=False, use_scale=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x),
            shift, scale
        )
        x = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=constant(0), dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a ViT backbone.
    """
    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 2
    learn_sigma: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t, y, train=True, force_drop_ids=None):
        """
        (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        """
        # sanity check dimensionality of t, y
        t = t.squeeze()
        y = y.squeeze()
        # overhead
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        out_channels = in_channels if not self.learn_sigma else in_channels * 2
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size
        pos_embed = self.variable(
            "state", "pos_embed", get_2d_sincos_pos_embed,
            self.hidden_size, num_patches, precision=self.param_dtype
        ).value.astype(self.dtype)
        # (B, num_patches, hidden_features)
        x = PatchEmbedder(self.patch_size, self.hidden_size,
                          dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = x + pos_embed
        # (B, hidden_features)
        t = TimestepEmbedder(self.hidden_size, dtype=self.dtype,
                             param_dtype=self.param_dtype)(t)
        # (B, hidden_features)
        y = LabelEmbedder(
            0, self.num_classes, self.hidden_size
        )(y, train=train, force_drop_ids=force_drop_ids)
        c = t + y
        for _ in range(self.depth):
            x = DiTBlock(
                self.hidden_size, self.num_heads, self.mlp_ratio,
                dtype=self.dtype, param_dtype=self.param_dtype,
            )(x, c, train=train)
        # (B, num_patches, p*p*c)
        x = FinalLayer(
            self.patch_size, out_channels, self.hidden_size,
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x, c)
        x = jnp.reshape(x, (batch_size, num_patches_side, num_patches_side,
                            self.patch_size, self.patch_size, out_channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C',
                      H=int(num_patches_side), W=int(num_patches_side))
        return x


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
