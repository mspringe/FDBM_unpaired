"""
This module provides a JAX implemention of various layers for network
architectures.


MultiHeadDotProductAttention has been adapred to feature cuDNN flash attention
(when possible), offering a significant boost in performance. However,
attention-dropout is omitted as it is currently not supported by cuDNN flash
attention.


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax
from jax.random import bernoulli
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from flax.linen.linear import default_kernel_init
from flax.linen.initializers import zeros, xavier_uniform, normal, constant
import warnings
import math
from einops import rearrange
import functools
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
FLASH_ATTENTION_WARNED = False


def qkv_flash_attention(q, k, v, *args, mask=None, **kwargs):
    """
    a simple QKV attention wrapper, that tries to use flash attention and falls
    back to the standard XLA implementation, when flash attention is not
    available

    Args:
        q: queries
        k: keys
        v: values
        mask: masking
    Returns:
        QKV attention of respective values
    Note: dropout arguments will be ignored
    """
    try:
        return jax.nn.dot_product_attention(
            q, k, v, implementation='cudnn', mask=mask
        )
    except Exception as e:
        global FLASH_ATTENTION_WARNED
        if not FLASH_ATTENTION_WARNED:
            warnings.warn(
                f'\n\033[93m'
                f'[User Warning]'
                f'\033[0m'
                f' cuDNN flash attention failed, falling back to XLA. '
                f'Stack-trace:\n'
                f'{e}',
                stacklevel=2
            )
            FLASH_ATTENTION_WARNED = True
        return jax.nn.dot_product_attention(
            q, k, v, implementation='xla', mask=mask
        )


def timestep_embedding(t, dim, max_period=10000, dtype=jnp.float32,
                       precision=jnp.float32):
    """
    creates sinusoidal timestep embeddings

    Args:
        t: a 1-D Tensor of N indices, one per batch element
        dim: the dimension of the output
        max_period: controls the minimum frequency of the embeddings
        dtype: data type of output
        precision: dtype of logs, exponentials and sinusoidals during
                   initialization of embeddings.
    Returns:
        positional embeddings, with shape (N, D)
    """
    t = t.astype(precision)
    half = dim // 2
    series = jnp.arange(start=0, stop=half, dtype=precision)
    freqs = jnp.exp(-math.log(max_period) * series / half)
    args = t[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    embedding = embedding.astype(dtype)
    return embedding


def pixel_shuffle(x, upscale_factor):
    """
    Rearranges elements in a tensor of shape (B, H, W, C*r^2) to (B, H*r, W*r, C)
    using the pixel shuffle operation.
    
    Args:
        x: Input tensor of shape (B, H, W, C * upscale_factor^2)
        upscale_factor: Integer factor by which spatial resolution is increased
    
    Returns:
        Tensor of shape (B, H * upscale_factor, W * upscale_factor, C)
    """
    r = upscale_factor
    return rearrange(x, 'b h w (c r1 r2) -> b (h r1) (w r2) c', r1=r, r2=r)


def pixel_unshuffle(x, downscale_factor):
    """
    Rearranges elements in a tensor of shape (B, H*r, W*r, C) to (B, H, W, C*r^2)
    using the pixel unshuffle operation.
    
    Args:
        x: Input tensor of shape (B, H * downscale_factor, W * downscale_factor, C)
        downscale_factor: Integer factor by which spatial resolution is reduced
    
    Returns:
        Tensor of shape (B, H, W, C * downscale_factor^2)
    """
    r = downscale_factor
    return rearrange(x, 'b (h r1) (w r2) c -> b h w (c r1 r2)', r1=r, r2=r)


class MultiHeadDotProductAttention(nn.Module):
    """
    Multi-head dot-product attention. Minor adaptations from:
    https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/attention.html#MultiHeadDotProductAttention

    Attributes:
        num_heads: number of attention heads. Features (i.e.
                   inputs_q.shape[-1]) should be divisible by the number of
                   heads.
        dtype: the dtype of the computation
               (default: infer from inputs and params)
        param_dtype: the dtype passed to parameter initializers
                     (default: float32)
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        broadcast_dropout: (Note: NOT USED) use a broadcasted dropout along
                           batch dims.
        dropout_rate: (Note: NOT USED) dropout rate
        deterministic: if false, the attention weight is masked randomly
                       using dropout, whereas if true, the attention weights
                       are deterministic.
        precision: numerical precision of the computation see
                   `jax.lax.Precision` for details.
        kernel_init: initializer for the kernel of the Dense layers.
        bias_init: initializer for the bias of the Dense layers.
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts
                      query, key, value, and returns output of shape
                      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`
        decode: whether to prepare and use an autoregressive cache.
    """
    num_heads: int
    dtype: Any = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    # broadcast_dropout: bool = True
    # dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[[Array, Array, Array], Array] = qkv_flash_attention
    decode: bool = False

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output
        vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(DenseGeneral,
                                  axis=-1,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype,
                                  features=(self.num_heads, head_dim),
                                  kernel_init=self.kernel_init,
                                  bias_init=self.bias_init,
                                  use_bias=self.use_bias,
                                  precision=self.precision)
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (dense(name='query')(inputs_q),
                             dense(name='key')(inputs_kv),
                             dense(name='value')(inputs_kv))
        query = jnp.asarray(query)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable('cache', 'cached_key',
                                       jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                                         jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                *batch_dims, max_length, num_heads, depth_head = (
                    cached_key.value.shape
                )
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_head)
                if expected_shape != query.shape:
                  raise ValueError('Autoregressive cache shape error, '
                                   'expected query shape %s instead got %s.' %
                                   (expected_shape, query.shape))
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value,
                                                 indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                                     tuple(batch_dims) + (1, 1, max_length))
                )
        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
        )
        # back to the original inputs dimensions
        out = DenseGeneral(features=features,
                           axis=(-2, -1),
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           precision=self.precision,
                           name='out')(x)
        return out


class TimestepEmbedder(nn.Module):
    """
    embeds scalar timesteps into vector representations
    """
    hidden_features: int
    frequency_embedding_size: int = 256
    max_period: int = 10000
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, t):
        sigma = 0.02
        x = timestep_embedding(t, self.frequency_embedding_size,
                               self.max_period, self.dtype,
                               precision=self.param_dtype)
        x = nn.Dense(self.hidden_features, kernel_init=normal(sigma),
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_features, kernel_init=normal(sigma),
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class LabelEmbedder(nn.Module):
    """
    embeds class labels into vector representations. Also handles label dropout
    for classifier-free guidance
    """
    dropout_prob: float
    num_classes: int
    hidden_features: int

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            rng = self.make_rng('dropout')
            drop_ids = bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        sigma = 0.02
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_features,
                                   embedding_init=normal(sigma))
        has_dropout = self.dropout_prob > 0
        if (train and has_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings


class MlpBlock(nn.Module):
    out_features: int = None
    hidden_features: int = None
    drop: float = 0.15
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        sigma = 0.02
        out_features = self.out_features or x.shape[-1]
        hidden_features = self.hidden_features or x.shape[-1]
        x = nn.Dense(
            features=hidden_features, kernel_init=xavier_uniform(),
            bias_init=normal(sigma), dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(
            features=out_features, kernel_init=xavier_uniform(),
            bias_init=normal(sigma),  dtype=self.dtype, param_dtype=self.dtype
        )(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        return x


class PatchEmbedder(nn.Module):
    patch_size: int
    hidden_features: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch = (self.patch_size, self.patch_size)
        num_H_patches = (H // self.patch_size)
        num_W_patches = (W // self.patch_size)
        x = nn.Conv(self.hidden_features, patch, patch, padding="VALID",
                    kernel_init=xavier_uniform(), dtype=self.dtype,
                    param_dtype=self.param_dtype)(x)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_H_patches,
                      w=num_W_patches)
        return x
