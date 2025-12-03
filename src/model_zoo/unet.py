"""
This module provides a JAX implemention of the Common UNet with scale, shift
conditioning on some encoding of time and classes.

Pixel-Wise attention for Feature-Map encodings can be used like in all UNet
implementations for diffusion-based models.

This implementation features cuDNN flash attention (when possible).


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, xavier_uniform
from typing import Any, Tuple
from einops import rearrange
from . import layers


def scale_shift(x, shift, scale, gate=None):
    """
    simple scale-shift conditioning:
    gate * ( x * ( 1 + scale ) + shift )

    Args:
        x: input images or vectors
        shift: the shift / bias to apply to this operation
        scale: the scaling factor will be (1+scale)
        gate: (optional) an additional scaling of the scaled x plus bias
    Returns:
        conditioned input via scaling and shift
    """
    # matching dimensionality
    add_dims = len(x.shape) - len(shift.shape)
    for _ in range(add_dims):
        scale = scale[:, None]
        shift = shift[:, None]
        if gate is not None:
            gate = gate[:, None]
    # applying scale shift (with gate)
    gate = 1 if gate is None else gate
    return gate * (x * (1 + scale) + shift)


class AttentionImg(nn.Module):
    """
    Pixel-wise self attention, where feature-maps are vectorized for the
    attention operator.
    """
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c, train=True):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
            c: conditioning encoding
            train: flag indicating the train/ eval. mode of Dropout
        Returns:
            scale-shifted pixel-wise self attention
        """
        B, H, W, C = x.shape
        # map to vector representation
        x = rearrange(x, 'b h w c -> b (h w) c')
        # attention
        qkv = nn.Dense(
            3*C, kernel_init=constant(0.0),
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = layers.qkv_flash_attention(q, k, v)
        # scale shift
        x = ScaleShift(self.dtype, self.param_dtype)(x, c, train)
        # map back to feature-map representation
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        return x


class ScaleShift(nn.Module):
    """A module-wrapper for the scale-shift function to simplify usage"""
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    p_drop: float = 0.15

    @nn.compact
    def __call__(self, x, c, train=True):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
            c: conditioning encoding
            train: flag indicating the train/ eval. mode of Dropout
        Returns:
            x scale-shifted for respective conditionals c
        """
        D = x.shape[-1]
        vals = layers.MlpBlock(
            out_features=3*D, hidden_features=D, drop=self.p_drop,
            dtype=self.dtype, param_dtype=self.param_dtype
        )(c, train=train)
        scale, shift, gate = jnp.split(vals, 3, axis=-1)
        x = x + scale_shift(x, shift, scale, gate)
        return x


class GroupNorm(nn.Module):
    """
    Sanitized version of GroupNorm that automitically detects when the number
    of groups is incompatible.
    """
    num_groups: int = 32
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
        Returns:
            normalized x
        """
        C = x.shape[-1]
        num_groups = self.num_groups
        while C % num_groups != 0:
            num_groups //= 2
        return nn.GroupNorm(num_groups, dtype=self.dtype,
                            param_dtype=self.param_dtype)(x)


class Upsample(nn.Module):
    """Upsampling via convolution and pixle shuffle."""
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    norm: Any = GroupNorm
    factor: int = 2

    @nn.compact
    def __call__(self, x):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
        Returns:
            x upsampled by a factor of 2
        """
        dim_out = self.factor**2 * x.shape[-1]
        x = self.norm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=dim_out, kernel_size=(3, 3),
            dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=xavier_uniform()
        )(x)
        x = layers.pixel_shuffle(x, self.factor)
        return x


class Downsample(nn.Module):
    """Downsampling via pixel unshuffle."""
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    norm: Any = GroupNorm
    factor: int = 2

    @nn.compact
    def __call__(self, x):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
        Returns:
            x downsampled by a factor of 2
        """
        dim_out = x.shape[-1]
        x = self.norm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.silu(x)
        x = layers.pixel_unshuffle(x, self.factor)
        x = nn.Conv(
            features=dim_out, kernel_size=(3, 3),
            dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=xavier_uniform()
        )(x)
        return x


class ResBlock(nn.Module):
    """Simple residual block with scale-shift conditioning."""
    dim_out: int = None
    norm: Any = GroupNorm
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c, train=True):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
            c: conditioning encoding
            train: flag indicating the train/ eval. mode of Dropout
        Returns:
            sum of inputs and residuals, where residuals have been 
            scale-shifted w.r.t. c
        """
        dim_out = x.shape[-1] if self.dim_out is None else self.dim_out
        x_act = self.norm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x_act = nn.silu(x_act)
        out = nn.Conv(
            features=dim_out, kernel_size=(3, 3),
            dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=xavier_uniform()
        )(x_act)
        out = ScaleShift(self.dtype, self.param_dtype)(out, c, train)
        x = nn.Conv(
            features=dim_out, kernel_size=(1, 1),
            dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=xavier_uniform()
        )(x_act)
        x = x + out
        return x


class ResBlockSeq(nn.Module):
    """A sequence of residual blocks."""
    dim_out: int = None
    num_res_blocks: int = 3
    norm: Any = GroupNorm
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c, train=True, res=None):
        """
        Applies a sequence of residual blocks, where each residual block may
        receive an residual input. This allows for longer squences of residual
        blocks, because of the block-wise skip connections and gradient
        propagation.
        (Credit to github.com/openai/consistency_models/blob/main/cm/unet.py)

        Args:
            x: jax.numpy tensor of images with channel-last convention
            c: conditioning encoding
            train: flag indicating the train/ eval. mode of Dropout
            res: (optional) a list of residuals for respective blocks.
                 Note: the length of `res` has to match `num_res_blocks`
        Returns:
            output of residual-block sequence
            and residual outputs (if res is None)
        """
        # if there are no residuals, we are in the downsampling stage, hence
        # residuals will be stored and returned
        if res is None:
            res_out = []
            for _ in range(self.num_res_blocks):
                # applying residual block
                x = ResBlock(self.dim_out, self.norm, self.dtype,
                             self.param_dtype)(x, c, train)
                # storing residual outputs
                res_out.append(x)
        # if there are residuals, we are in the upsampling stage, hence
        # no storing of residuals required
        else:
            res_out = None
            for i in range(self.num_res_blocks):
                # fetching and concatenating features from residual connection
                residual = res[i]
                x = jnp.concatenate([x, residual], axis=-1)
                # applying residual block
                x = ResBlock(self.dim_out, self.norm, self.dtype,
                             self.param_dtype)(x, c, train)
        return x, res_out


class UNet(nn.Module):
    """UNet with residual blocks and attention."""
    channels_out: int = None
    base_features: int = 128
    channel_mults: Tuple = (1, 2, 4)
    attention_levels: Tuple = (False, False, False)
    num_res_blocks: int = 3
    num_classes: int = 2
    hidden_size: int = 512
    norm: Any = GroupNorm
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    class_dropout_prob: float = 0

    @nn.compact
    def __call__(self, x, t, y, train=True, force_drop_ids=None):
        """

        Args:
            x: jax.numpy tensor of images with channel-last convention
            t: point in time from the interval (0, 1)
            y: integer, representation of some class or direction
            train: flag indicating the train/ eval. mode of Dropout
            force_drop_ids: explicit Label-Dropout ids
        Returns:
            network output without normalization or activation
        """
        channels_out = self.channels_out if self.channels_out is not None else x.shape[-1]
        # timestep embedding via sinusoidals
        t = layers.TimestepEmbedder(self.hidden_size, dtype=self.dtype,
                                    param_dtype=self.param_dtype)(t)
        # label embedding via embedding table and label-dropout
        y = layers.LabelEmbedder(
            self.class_dropout_prob, self.num_classes, self.hidden_size
        )(y, train=train, force_drop_ids=force_drop_ids)
        # combining condtioning via addition (`hidden_size` stays consistent)
        c = t + y
        # additional inputs (conditioning context, train state)
        meta = (c, train)
        # first layer
        x = ResBlock(self.channel_mults[0]*self.base_features, self.norm,
                     self.dtype, self.param_dtype)(x, *meta)
        # down
        residuals = []
        down_args = tuple(enumerate(zip(
            self.channel_mults, 
            self.attention_levels
        )))
        for lvl, (cmul, use_attn) in down_args:
            # dimensionality of current level
            dim_out = self.base_features * cmul
            # residual blocks
            x, res = ResBlockSeq(dim_out, self.num_res_blocks, self.norm, 
                                 self.dtype, self.param_dtype)(x, *meta)
            residuals.append(res)
            # attention operator
            if use_attn:
                x = AttentionImg(self.dtype, self.param_dtype)(x, *meta)
            # downsampling
            if (lvl + 1) < len(down_args):  # no downsampling on the last level
                x = Downsample(self.dtype, self.param_dtype)(x)
        # bottleneck with attention
        x = ResBlock(dim_out, self.norm, self.dtype, self.param_dtype)(x, *meta)
        x = AttentionImg(self.dtype, self.param_dtype)(x, *meta)
        x = ResBlock(dim_out, self.norm, self.dtype, self.param_dtype)(x, *meta)
        # up
        up_args = tuple(enumerate(zip(
            self.channel_mults[::-1],
            residuals[::-1],
            self.attention_levels[::-1]
        )))
        for lvl, (cmul, res, use_attn) in up_args:
            # upsampling
            if lvl > 0:  # no downsampling on the last (bottleneck-) level
                x = Upsample(self.dtype, self.param_dtype)(x)
            # dimensionality of current level
            dim_out = self.base_features * cmul
            # residual blocks
            res = res[::-1]
            x, _ = ResBlockSeq(dim_out, self.num_res_blocks, self.norm, 
                               self.dtype, self.param_dtype)(x, *meta, res)
            # attention operator
            if use_attn:
                x = AttentionImg(self.dtype, self.param_dtype)(x, *meta)
        # final layer
        x = ResBlock(channels_out, self.norm, self.dtype,
                     self.param_dtype)(x, *meta)
        return x


def UNet_S_32x32(**kwargs):
    return UNet(
        base_features=32,
        channel_mults=(1, 2, 4),
        attention_levels=[False, False, False],
        num_res_blocks=2,
        **kwargs
    )


def UNet_B_32x32(**kwargs):
    return UNet(
        base_features=128,
        channel_mults=(1, 2, 4),
        attention_levels=[False, False, False],
        num_res_blocks=4,
        **kwargs
    )


def UNet_L_32x32(**kwargs):
    return UNet(
        base_features=128,
        channel_mults=(1, 2, 4),
        attention_levels=[False, True, True],
        num_res_blocks=3,
        **kwargs
    )


def UNet_XL_32x32(**kwargs):
    return UNet(
        base_features=196,
        channel_mults=(1, 2, 4),
        attention_levels=[False, True, True],
        num_res_blocks=4,
        **kwargs
    )


def UNet_S_64x64(**kwargs):
    return UNet(
        base_features=32,
        channel_mults=(1, 2, 2, 4),
        attention_levels=[False, False, False, False],
        num_res_blocks=2,
        **kwargs
    )


def UNet_B_64x64(**kwargs):
    return UNet(
        base_features=128,
        channel_mults=(1, 2, 3, 4),
        attention_levels=[False, False, False, False],
        num_res_blocks=4,
        **kwargs
    )


def UNet_L_64x64(**kwargs):
    return UNet(
        base_features=128,
        channel_mults=(1, 2, 4, 8),
        attention_levels=[False, False, True, True],
        num_res_blocks=3,
        **kwargs
    )


def UNet_XL_64x64(**kwargs):
    return UNet(
        base_features=196,
        channel_mults=(1, 2, 4),
        attention_levels=[False, True, True],
        num_res_blocks=4,
        **kwargs
    )


def UNet_XXL_64x64(**kwargs):
    return UNet(
        base_features=228,
        channel_mults=(1, 2, 4),
        attention_levels=[False, True, True],
        num_res_blocks=4,
        **kwargs
    )


UNet_models = {
    'UNet-XXL/64': UNet_XXL_64x64,
    'UNet-XL/32': UNet_XL_32x32, 'UNet-XL/64': UNet_XL_64x64,
    'UNet-L/32':  UNet_L_32x32,  'UNet-L/64':  UNet_L_64x64,
    'UNet-B/32':  UNet_B_32x32,  'UNet-B/64':  UNet_B_64x64,
    'UNet-S/32':  UNet_S_32x32,  'UNet-S/64':  UNet_S_64x64,
}
