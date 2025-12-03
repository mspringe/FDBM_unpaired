r"""
This module provides a JAX implemention of alpha-diffusion training, as
proposed by Bortoli et al. 2024 (see arXiv:2409.09347).


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import os
from functools import partial
import optax
import jax
import jax.numpy as jnp
from jax.random import uniform, normal
import jax.tree_util as jtu
import flax.linen as nn
from flax.training.train_state import TrainState
import math
from einops import rearrange
from tqdm import tqdm
import pickle
from ..opt_lib.loss_functions import mse, step
from ..opt_lib.ema import update_ema
from .. import distributed as dist
from .. import mixed_precision as mp
from .. import data
from .fsb import sample_pinned, input_transform, cond_var
from .loss import make_target
from .solver import em_fwd
from flax import serialization
from flax.training import checkpoints
from flax.serialization import from_state_dict


def select_t_max(fsb, verbose=True):
    steps = 10000
    start_from = 0.7
    threshold = 0.01
    for t in jnp.linspace(start_from, fsb.T, steps):
        # sigma_{T|t}
        sigma_Tt = cond_var(t, fsb.T, fsb.omega, fsb.gamma, fsb.g_max)
        if sigma_Tt < threshold:
            if verbose:
                # highlighting blue
                blue = '\033[94m'
                neutral = '\033[0m'
            break
    return t


def copy_params(params):
    return serialization.from_state_dict(params,
                                         serialization.to_state_dict(params))


def write_checkpoint(log_dir, step, pfx='', keep_max=3, verbose=True, **ckpt):
    ckpt_pth = checkpoints.save_checkpoint(
        log_dir, step=step, prefix=pfx, keep=keep_max, target=ckpt
    )
    if verbose:
        # highlighting blue
        blue = '\033[94m'
        neutral = '\033[0m'
        print(f'{blue}[CKPT]{neutral} wrote checkpoint to {ckpt_pth}',
              flush=True)


def load_checkpoint(ckpt_pth, state, verbose=True,
                    stage='pretraining'):
    ckpt = checkpoints.restore_checkpoint(ckpt_pth, target=None)
    states = from_checkpoint(ckpt, state, stage)
    if verbose:
        # highlighting blue
        blue = '\033[94m'
        neutral = '\033[0m'
        print(f'{blue}[CKPT]{neutral} loaded checkpoint from {ckpt_pth}',
              flush=True)
    return states


def from_checkpoint(ckpt, state, stage='pretraining'):
    # base parameters
    opt_state = from_state_dict(state.opt_state, ckpt['opt_state'])
    state = TrainState(
      step=ckpt['state']['step'],
      apply_fn=state.apply_fn,
      params=ckpt['state']['params'],
      tx=state.tx,
      opt_state=opt_state,
    )
    ema_parameters = [ckpt['ema_parameters'][k]
                      for k in sorted(ckpt['ema_parameters'].keys())]
    ema_rates = [ckpt['ema_rates'][k] for k in sorted(ckpt['ema_rates'].keys())]
    key_noise_fwd = ckpt['key_noise_fwd']
    key_noise_bwd = ckpt['key_noise_bwd']
    key_time = ckpt['key_time']
    key_dropout = ckpt['key_dropout']
    # pretrtaining only needs base parameters
    if stage == 'pretraining':
        return (
            state,
            ema_parameters, ema_rates,
            key_noise_fwd, key_noise_bwd,
            key_time, key_dropout
        )
    # finetuning needs additional PRNGKeys for SDEs
    elif stage == 'finetuning':
        key_sde_fwd = ckpt['key_sde_fwd']
        key_sde_bwd = ckpt['key_sde_bwd']
        return (
            state,
            ema_parameters, ema_rates,
            key_noise_fwd, key_noise_bwd,
            key_time, key_dropout,
            key_sde_fwd, key_sde_bwd
        )
    else:
        raise ValueError(f'unknown stage: "{stage}"; choose from '
                         f'{{pretraining, finetuning}}')


def assert_sanity(batch_size, micro_batch_size, stage=None):
    # sanity checks w.r.t. batch sizes
    assert batch_size >= micro_batch_size, \
           'the batch size should be greater or equal to the micro batch size'
    assert micro_batch_size >= 8, 'micro batch size is unreasonably small'
    assert micro_batch_size % dist.DEV_COUNT == 0, \
           'the batch size should be divisible by number of devices'
    assert batch_size % micro_batch_size == 0, \
           'the batch size should be divisible by micro-batch size'
    if stage == 'finetuning':
        assert micro_batch_size % 4 == 0, \
               'micro batch size should be divisible by 4'
    else:
        assert micro_batch_size % 2 == 0, \
               'micro batch size should be divisible by 2'


def log_step(logger, stp, apply_fn, fsb, params, X_0, X_1, directions, mean_loss,
             stage=None, solver=None, solver_steps=100, pi_0=None, pi_1=None,
             **log_step_fns):
    # starting points
    X = jnp.concatenate([X_0, X_1], axis=0)
    X = jax.device_put(X, dist.DP_SHARDING)
    _1 = jnp.ones(len(X_0), dtype=jnp.int32)
    _0 = jnp.zeros(len(X_1), dtype=jnp.int32)
    # endpoints
    Y_0 = 0
    # forward
    zt_fwd = em_fwd(apply_fn, params, fsb, X_0, _1, num_steps=solver_steps, Y_0=Y_0)
    imgs_forwd = zt_fwd[..., 0]
    # backward
    zt_bwd = em_fwd(apply_fn, params, fsb, X_1, _0, num_steps=solver_steps, Y_0=Y_0)
    imgs_backwd = zt_bwd[..., 0]
    # recover from latent space (if needs be)
    if isinstance(pi_0, data.LatentDataset):
        imgs_backwd = pi_0.decode(imgs_backwd[:8], device='cpu')
    else:
        imgs_backwd = pi_0.unscale(imgs_backwd)
    if isinstance(pi_1, data.LatentDataset):
        imgs_forwd = pi_1.decode(imgs_forwd[:8], device='cpu')
    else:
        imgs_forwd = pi_1.unscale(imgs_forwd)
    # log
    logger.log_scalar('optimization steps', stp, log_section=stage)
    logger.log_scalar('loss', mean_loss, log_section=stage)
    for k in log_step_fns.keys():
        logger.log_scalar(k, log_step_fns[k](stp), log_section=stage)
    logger.log_images('samples forward', imgs_forwd, log_section=stage)
    logger.log_images('samples backward', imgs_backwd, log_section=stage)
    if not isinstance(pi_0, data.LatentDataset) and \
       not isinstance(pi_1, data.LatentDataset):
        unscld_X = jnp.concatenate([pi_0.unscale(X_0), pi_1.unscale(X_1)],
                                   axis=0)
        logger.log_images('inputs', unscld_X, log_section=stage)


def resolve_acc_steps(bs, mbs):
    assert bs >= mbs, "batch can't be smaller than micro-batch"
    acc_steps = bs / mbs
    acc_steps = int(acc_steps) + int(acc_steps % 1.0 != 0)
    return acc_steps


def time_sampling(key, n, dtype=jnp.float32, t_offset=0.05, t_max=None):
    r"""
    Args:
        key: PRNGKey for sampling
        n: number of samples
        dtype: (optional) the data-type of time values (e.g. float32, bfloat16)
        t_offset: (optional) a small number, scaling the uniform distribution to
                  [t_offset, 1-t_offset]. Useful for numeric stability in the
                  loss.
        t_max: (optional) a number near 1, scaling the uniform distribution to
               [t_offset, t_max]. Useful for numeric stability in the loss.
               Overrides max value from t_offset if not None.
    Returns:
        samples in range [t_offset, 1-t_offset], with shape (n, 1)
    """
    if t_max is None:
        t_max = 1 - t_offset
    return uniform(key, (n, 1), dtype=dtype, minval=t_offset, maxval=t_max)


def noise_sampling(key, shape, dtype=jnp.float32):
    r"""
    Args:
        key: PRNGKey for sampling
        shape: shape of samples
        dtype: (optional) the data-type of time values (e.g. float32, bfloat16)
    Returns:
        samples of a standard normal distribution with zero mean and unit
        variance.
    """
    return normal(key, shape, dtype=dtype)


def cast2d(arr_1d):
    r"""
    Args:
        arr_1d: any array or tensor
    Returns:
        array with 3 new, appended dimensions to an (1D) array, allowing
        operations with images.
    """
    return arr_1d.squeeze()[:,None,None,None]


def pretraining(
        fsb,
        model,
        model_parameters,
        ema_parameters,
        ema_rates,
        optimizer,
        pi_0,
        pi_1,
        batch_size,
        micro_batch_size,
        eps=1.0,
        log_interval=int(1e4),
        checkpoint_interval=int(5e4),
        loss_fn=mse,
        dtype=jnp.float32,
        rngs=None,
        progbar=False,
        seed_time=3,
        seed_noise=42,
        pretraining_steps=int(1e5),
        log_dir='.',
        logger=None,
        log_step_fns={},
        mp_policy=mp.DEFAULT_POLICY,
        loss_scaler=mp.DEFAULT_LOSS_SCALER,
        ckpt_pretraining=None,
        keep_max=3
    ):
    r"""
    Args:
        model: a flax model
        model_parameters: parameterization of model
        ema_parameters: list of EMA parameters
        ema_rates: list of EMA rates for respective EMA parameters
        optimizer: an optax optimizer (e.g. `optax.radam`)
        pi_0: a `jax_lib.data.Dataset`
        pi_1: a `jax_lib.data.Dataset`
        batch_size: size of the mini-batch
        micro_batch_size: size of the microbatch, i.e. a fraction of the
                          mini-batch, used for gradient accumulation
        eps: entropic regulatization parameter
        log_interval: intermediate results will be logged every `log_interval`
                      steps
        checkpoint_interval: checkpoiunts will be saved every
                             `checkpoint_interval` steps
        loss_fn: loss function to be used
        dtype: data type for computation (weights will allways be stored as
               float32)
        rngs: random number generators for e.g. dropout
        progbar: flag indicating whethre to print a progress bar
        seed_time: the initial seed for RNGs of time-step samplings
        seed_noise: the initial seed for RNGs of noise samplings
        pretraining_steps: number of steps to pretrain
        log_dir: directory where logging files & checkpoints will be stored
        logger: Any instance of `jax_lib.log_lib.Logger`
        log_step_fns: a dictionary of functions that take the step-count and 
                      return a value to be logged (e.g. lr-schedule)
        log_dir: directory where logging files & checkpoints will be stored
        mp_policy: mixed-precision policy of compute-, parameter-, and output
                   data type
    Returns:
        the model, model parameters, EMA parameters, and training state
    """
    # skipping training, if no steps specified
    if pretraining_steps is None or pretraining_steps <= 0:
        return model, model_parameters, ema_parameters, None

    # sanity checks
    assert_sanity(batch_size, micro_batch_size)

    # scale pretraining steps to a micro-batch perspective
    acc_steps = resolve_acc_steps(batch_size, micro_batch_size)
    pretraining_steps = pretraining_steps * acc_steps

    # seeds for time- and noise samplers
    key_time = jax.random.PRNGKey(seed_time)
    key_noise_fwd = jax.random.PRNGKey(seed_noise)
    key_noise_bwd = jax.random.PRNGKey(seed_noise+1)

    # flax training state initialization
    state = TrainState.create(apply_fn=model.apply, params=model_parameters,
                              tx=optimizer)

    # half micro batch size for splitting batches w.r.t. fwd & bwd SDEs
    half_mb = micro_batch_size // 2
    # direction-indices for fwd & bwd SDEs
    _1 = jnp.ones((half_mb, 1), dtype=jnp.int32)   # 1 <- fwd
    _0 = jnp.zeros((half_mb, 1), dtype=jnp.int32)  # 0 <- bwd
    directions = jnp.concatenate([_1, _0], axis=0)
    directions = jax.device_put(directions, dist.DP_SHARDING)
    T = jax.device_put(jnp.ones([micro_batch_size]), dist.DP_SHARDING)

    # direct reference for dropout PRNG
    key_dropout = rngs['dropout']

    # load from checkpoint
    if ckpt_pretraining is not None:
        ckpt_pretraining = os.path.abspath(ckpt_pretraining)
        (state,
         ema_parameters,
         key_noise_fwd, key_noise_bwd,
         key_time, key_dropout) = load_checkpoint(ckpt_pretraining,
                                                  state,
                                                  stage='pretraining')
        it_start = state.step
    else:
        it_start = 0

    # logging setup & running mean loss
    mean_loss = 0
    steps_range = jnp.arange(it_start, pretraining_steps)
    if progbar:
        iterator = tqdm(steps_range)
    else:
        iterator = steps_range

    # Pretraining loop
    TMAX = select_t_max(fsb)
    for iteration in iterator:
        # (I) Sampling from data
        # sample from terminal distributions
        X_0, _ = pi_0.sample(micro_batch_size)
        X_1, _ = pi_1.sample(micro_batch_size)
        # new PRNG states
        key_dropout, subkey_dropout = jax.random.split(key_dropout)
        key_time, subkey_time = jax.random.split(key_time)
        key_noise_fwd, subkey_noise_fwd = jax.random.split(key_noise_fwd)
        key_noise_bwd, subkey_noise_bwd = jax.random.split(key_noise_bwd)
        # distributed PRNG states
        dropout_rngs = jax.device_put(subkey_dropout, dist.REPLICATE_SHARDING)
        # input sharding
        X_0, X_1 = dist.map_sharding(dist.DP_SHARDING, X_0, X_1)
        ref_X_0 = jnp.concatenate([X_0[half_mb:], X_1[:half_mb]], axis=0)
        ref_X_1 = jnp.concatenate([X_1[half_mb:], X_0[:half_mb]], axis=0)
        ref_X_0, ref_X_1, _1, _0 = dist.map_sharding(
            dist.DP_SHARDING, ref_X_0, ref_X_1, _1, _0
        )

        # (II) Sampling from marginals of paths
        # sample time
        t = time_sampling(subkey_time, micro_batch_size, dtype=dtype,
                          t_max=TMAX)
        # sharding (for distributed training setup, see jax_lib.distributed)
        t = jax.device_put(t, dist.DP_SHARDING)
        # sample from path (reciprocal interpolation between terminals)
        Y_0 = 0
        Z_t = sample_pinned(
            subkey_noise_fwd, t, T, ref_X_0, ref_X_1,
            fsb.omega, fsb.gamma, fsb.g_max,
            Y_0=Y_0, channel_last=True
        )
        X_t = input_transform(Z_t[..., 0], Z_t[..., 1:],
                              t, jnp.ones_like(t), fsb.omega, fsb.gamma, fsb.g_max,
                              direction=1)
        # assign targets to respective forw & back paths
        targets = make_target(ref_X_1, Z_t[..., 0], Z_t[..., 1:],
                         t, jnp.ones_like(t),
                         fsb.omega, fsb.gamma, fsb.g_max, direction=1)  # fwd

        # (III) Optimization step
        X_t, targets, t = dist.map_sharding(dist.DP_SHARDING, X_t, targets, t)
        # optimization step
        loss, state, _, loss_scaler = step(
            state, X_t, targets, t, directions,
            dropout_rngs, eps=eps,
            mp_policy=mp_policy, loss_scaler=loss_scaler
        )
        # updating EMA after every step
        if (iteration+1) % float(acc_steps) == 0:
            ema_parameters = [update_ema(ep, state.params, er)
                              for ep, er in zip(ema_parameters, ema_rates)]

        # (IV) Logging & saving weights
        mean_loss += loss / (log_interval * acc_steps)
        stp = (iteration+1) / acc_steps
        # logging progress
        if stp % log_interval == 0 and logger is not None:
            # decoding from latent space (note that latent space is normalized)
            log_step(logger, int(stp), state.apply_fn, fsb, ema_parameters[0],
                     X_0[:half_mb], X_1[:half_mb], directions, mean_loss,
                     stage='pretraining', pi_0=pi_0, pi_1=pi_1,
                     **log_step_fns)
            mean_loss = 0
        # saving checkpoints
        if stp % checkpoint_interval == 0:
            keys = dict(
                key_noise_fwd=key_noise_fwd,
                key_noise_bwd=key_noise_bwd,
                key_time=key_time,
                key_dropout=key_dropout
            )
            ckpt = dict(
                state=state,
                opt_state=serialization.to_state_dict(state.opt_state),
                ema_parameters=ema_parameters,
                ema_rates=ema_rates,
                **keys
            )
            write_checkpoint(
                os.path.abspath(log_dir), int(stp), pfx='pretraining', keep_max=keep_max, verbose=True,
                **ckpt
            )
    return model, (state.params, ema_parameters, state)


def finetuning(
        fsb,
        model,
        model_parameters,
        ema_parameters,
        ema_rates,
        optimizer,
        pi_0,
        pi_1,
        batch_size,
        micro_batch_size,
        eps=1.0,
        log_interval=int(1e4),
        checkpoint_interval=int(5e4),
        loss_fn=mse,
        dtype=jnp.float32,
        rngs=None,
        progbar=False,
        seed_time=3,
        seed_noise=42,
        finetuning_steps=int(1e5),
        log_dir='.',
        logger=None,
        log_step_fns={},
        state=None,
        solver_steps=40,
        solver=None,
        mp_policy=mp.DEFAULT_POLICY,
        loss_scaler=mp.DEFAULT_LOSS_SCALER,
        ckpt_finetuning=None,
        keep_max=3
    ):
    r"""
    Args:
        model: a flax model
        model_parameters: parameterization of model
        ema_parameters: list of EMA parameters
        ema_rates: list of EMA rates for respective EMA parameters
        optimizer: an optax optimizer (e.g. `optax.radam`)
        pi_0: a `jax_lib.data.Dataset`
        pi_1: a `jax_lib.data.Dataset`
        batch_size: size of the mini-batch
        micro_batch_size: size of the microbatch, i.e. a fraction of the
                          mini-batch, used for gradient accumulation
        eps: entropic regulatization parameter
        log_interval: intermediate results will be logged every `log_interval`
                      steps
        checkpoint_interval: checkpoiunts will be saved every
                             `checkpoint_interval` steps
        loss_fn: loss function to be used
        dtype: data type for computation (weights will always be stored as
               float32)
        rngs: random number generators for e.g. dropout
        progbar: flag indicating whethre to print a progress bar
        seed_time: the initial seed for RNGs of time-step samplings
        seed_noise: the initial seed for RNGs of noise samplings
        finetuning_steps: number of steps to finetune
        log_dir: directory where logging files & checkpoints will be stored
        logger: Any instance of `jax_lib.log_lib.Logger`
        log_step_fns: a dictionary of functions that take the step-count and 
                      return a value to be logged (e.g. lr-schedule)
        state: flax `TrainState` from pretraining
        solver_steps: number of solver discretization-steps / function evals
        solver: some SDE/ ODE solver, deafaults to Euler-Maruyama
                (see `jax_lib.sampling_lib.solvers` for options)
        log_dir: directory where logging files & checkpoints will be stored
        mp_policy: mixed-precision policy of compute-, parameter-, and output
                   data type
    Returns:
        the model, model parameters, EMA parameters, and training state
    """
    # skipping training, if no steps specified
    if finetuning_steps is None or finetuning_steps <= 0:
        return model, model_parameters, ema_parameters, None

    # sanity checks
    assert_sanity(batch_size, micro_batch_size)

    # scale pretraining steps to a micro-batch perspective
    acc_steps = resolve_acc_steps(batch_size, micro_batch_size)
    pretraining_steps = finetuning_steps * acc_steps

    # seeds for time- and noise samplers
    key_time = jax.random.PRNGKey(seed_time)
    key_noise_fwd = jax.random.PRNGKey(seed_noise)
    key_noise_bwd = jax.random.PRNGKey(seed_noise+1)
    key_sde_fwd = jax.random.PRNGKey(seed_noise+2)
    key_sde_bwd = jax.random.PRNGKey(seed_noise+3)

    # flax training state initialization
    #if state is None:
    state = TrainState.create(apply_fn=model.apply, params=model_parameters,
                              tx=optimizer)

    # half micro batch size for splitting batches w.r.t. fwd & bwd SDEs
    half_mb = micro_batch_size // 2
    # direction-indices for fwd & bwd SDEs
    _1 = jnp.ones((half_mb, 1), dtype=jnp.int32)   # 1 <- fwd
    _0 = jnp.zeros((half_mb, 1), dtype=jnp.int32)  # 0 <- bwd
    directions = jnp.concatenate([_1, _0], axis=0)
    directions = jax.device_put(directions, dist.DP_SHARDING)
    T = jax.device_put(jnp.ones([2*half_mb]), dist.DP_SHARDING)

    # direct reference for dropout PRNG
    key_dropout = rngs['dropout']

    # load from checkpoint
    if ckpt_finetuning is not None:
        ckpt_finetuning = os.path.abspath(ckpt_finetuning)
        (state,
         ema_parameters, ema_rates,
         key_noise_fwd, key_noise_bwd,
         key_time, key_dropout,
         key_sde_fwd, key_sde_bwd) = load_checkpoint(ckpt_finetuning,
                                                     state,
                                                     stage='finetuning')
        it_start = state.step
    else:
        it_start = 0

    # logging setup & running mean loss
    mean_loss = 0
    steps_range = jnp.arange(it_start, pretraining_steps)
    if progbar:
        iterator = tqdm(steps_range)
    else:
        iterator = steps_range

    # Finetuning loop
    TMAX = select_t_max(fsb)
    for iteration in iterator:
        # (I) Sampling from data
        # sample from terminal distributions
        X_0, _ = pi_0.sample(micro_batch_size//2)
        X_1, _ = pi_1.sample(micro_batch_size//2)
        # new PRNG states
        key_dropout, subkey_dropout = jax.random.split(key_dropout)
        key_time, subkey_time = jax.random.split(key_time)
        key_noise_fwd, subkey_noise_fwd = jax.random.split(key_noise_fwd)
        key_noise_bwd, subkey_noise_bwd = jax.random.split(key_noise_bwd)
        key_sde_fwd, subkey_sde_fwd = jax.random.split(key_sde_fwd)
        key_sde_bwd, subkey_sde_bwd = jax.random.split(key_sde_bwd)
        # distributed PRNG states
        dropout_rngs = jax.device_put(subkey_dropout, dist.REPLICATE_SHARDING)
        # input sharding
        ref_X_1 = jnp.concatenate([X_1, X_0], axis=0)
        ref_X_1, _1, _0 = dist.map_sharding(
            dist.DP_SHARDING, ref_X_1, _1, _0
        )

        # endpoints
        Y_0 = 0
        # forward
        zt_fwd = em_fwd(state.apply_fn, ema_parameters[0], fsb, X_0, _1,
                        num_steps=100, Y_0=Y_0, key=subkey_sde_fwd)
        imgs_forwd = zt_fwd[..., 0]
        imgs_forwd = pi_1.scale(pi_1.unscale(imgs_forwd).clip(-1, 1))
        # backward
        zt_bwd = em_fwd(state.apply_fn, ema_parameters[0], fsb, X_1,
                        _0, num_steps=100, Y_0=Y_0, key=subkey_sde_bwd)
        imgs_backwd = zt_bwd[..., 0]
        imgs_backwd = pi_0.scale(pi_0.unscale(imgs_backwd).clip(-1, 1))
        imgs = jnp.concatenate([imgs_backwd, imgs_forwd], axis=0)

        # (II) Sampling from marginals of paths
        # sample time
        t = time_sampling(subkey_time, micro_batch_size, dtype=dtype,
                          t_max=TMAX)
        t_fwd = t[:half_mb]
        t_bwd = t[half_mb:]
        # sharding (for distributed training setup, see jax_lib.distributed)
        t, imgs = dist.map_sharding(dist.DP_SHARDING, t, imgs)
        # sample from path (reciprocal interpolation between terminals)
        Y_0 = 0
        Z_t= sample_pinned(
            subkey_noise_fwd, t, T, imgs, ref_X_1,
            fsb.omega, fsb.gamma, fsb.g_max,
            Y_0=Y_0, channel_last=True
        )
        X_t = input_transform(Z_t[..., 0], Z_t[..., 1:],
                              t, jnp.ones_like(t), fsb.omega, fsb.gamma, fsb.g_max,
                              direction=1)
        # assign targets to respective forw & back paths
        targets = make_target(ref_X_1, Z_t[..., 0], Z_t[..., 1:],
                         t, jnp.ones_like(t),
                         fsb.omega, fsb.gamma, fsb.g_max, direction=1)  # fwd

        # (III) Optimization step
        X_t, targets = dist.map_sharding(dist.DP_SHARDING, X_t, targets)
        # optimization step
        loss, state, _, loss_scaler = step(
            state, X_t, targets, t, directions,
            dropout_rngs, eps=eps,
            mp_policy=mp_policy, loss_scaler=loss_scaler
        )
        # updating EMA after every step
        if (iteration+1) % float(acc_steps) == 0:
            ema_parameters = [update_ema(ep, state.params, er)
                              for ep, er in zip(ema_parameters, ema_rates)]

        # (IV) Logging & saving weights
        mean_loss += loss / (log_interval * acc_steps)
        stp = (iteration+1) / acc_steps
        # logging progress
        if stp % log_interval == 0 and logger is not None:
            # decoding from latent space (note that latent space is normalized)
            log_step(logger, int(stp), state.apply_fn, fsb, ema_parameters[0],
                     X_0[:half_mb], X_1[:half_mb], directions, mean_loss,
                     stage='finetuning', pi_0=pi_0, pi_1=pi_1,
                     **log_step_fns)
            mean_loss = 0
        # saving checkpoints
        if stp % checkpoint_interval == 0:
            keys = dict(
                key_noise_fwd=key_noise_fwd,
                key_noise_bwd=key_noise_bwd,
                key_time=key_time,
                key_dropout=key_dropout,
                key_sde_fwd=key_sde_fwd,
                key_sde_bwd=key_sde_bwd
            )
            write_checkpoint(
                os.path.abspath(log_dir), int(stp), pfx='finetuning', keep_max=keep_max, verbose=True,
                state=state,
                ema_parameters=ema_parameters,
                ema_rates=ema_rates,
                **keys
            )
    return model, (state.params, ema_parameters, state)


def train(
        fsb,
        model,
        model_parameters,
        ema_parameters,
        ema_rates,
        optimizer,
        lr_sched,
        pi_0,
        pi_1,
        batch_size,
        micro_batch_size,
        alpha,
        eps=1.0,
        log_interval=int(1e4),
        checkpoint_interval=int(5e4),
        loss_fn=mse,
        dtype=jnp.float32,
        rngs=None,
        progbar=False,
        seed_time=3,
        seed_noise=42,
        pretraining_steps=int(1e5),
        finetuning_steps=int(1e5),
        log_dir='.',
        logger=None,
        mp_policy=mp.DEFAULT_POLICY,
        loss_scaler=mp.DEFAULT_LOSS_SCALER,
        ckpt_pretraining=None,
        ckpt_finetuning=None,
        keep_max=3
    ):
    r"""
    Args:
        model: a flax model
        model_parameters: parameterization of model
        ema_parameters: list of EMA parameters
        ema_rates: list of EMA rates for respective EMA parameters
        optimizer: an optax optimizer constructor only the learning rate as its
                   only argument
        lr_sched: a learning rate schedule constructor, that only takes the
                  learning rate and total number of optimization steps as its
                  only arguments
        pi_0: a `jax_lib.data.Dataset`
        pi_1: a `jax_lib.data.Dataset`
        batch_size: size of the mini-batch
        micro_batch_size: size of the microbatch, i.e. a fraction of the
                          mini-batch, used for gradient accumulation
        alpha: alpha parameter of alpha-diffusion, which also represents the
               learning rate
        eps: entropic regulatization parameter
        log_interval: intermediate results will be logged every `log_interval`
                      steps
        checkpoint_interval: checkpoiunts will be saved every
                             `checkpoint_interval` steps
        loss_fn: loss function to be used
        dtype: data type for computation (weights will always be stored as
               float32)
        rngs: random number generators for e.g. dropout
        progbar: flag indicating whethre to print a progress bar
        seed_time: the initial seed for RNGs of time-step samplings
        seed_noise: the initial seed for RNGs of noise samplings
        pretraining_steps: number of steps to pretrain
        finetuning_steps: number of steps to finetune
        log_dir: directory where logging files & checkpoints will be stored
        mp_policy: mixed-precision policy of compute-, parameter-, and output
                   data type
        loss_scaler=mixed_precision.DEFAULT_LOSS_SCALER
    Returns:
        the model, model parameters, EMA parameters, and training state
    """
    state = None
    # pretraining loop
    if pretraining_steps > 0:
        lr_schedule = lr_sched(alpha, pretraining_steps)
        log_step_fns = {'lr-schedule': lr_schedule,
                        'loss-scale': lambda _ : loss_scaler.loss_scale}
        opt = optimizer(lr_schedule)
        model, (model_parameters, ema_parameters, state) = pretraining(
            fsb,
            model,
            model_parameters,
            ema_parameters,
            ema_rates,
            opt,
            pi_0,
            pi_1,
            batch_size,
            micro_batch_size,
            eps=eps,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            loss_fn=loss_fn,
            dtype=dtype,
            rngs=rngs,
            progbar=progbar,
            seed_time=seed_time,
            seed_noise=seed_noise,
            pretraining_steps=pretraining_steps,
            log_dir=log_dir,
            logger=logger,
            log_step_fns=log_step_fns,
            mp_policy=mp_policy,
            loss_scaler=loss_scaler,
            ckpt_pretraining=ckpt_pretraining
        )
    # finetuning loop
    if finetuning_steps > 0:
        # using half the learning rate for fine-tuning
        lr_schedule = lr_sched(alpha/2, finetuning_steps, warmup_steps=1000)
        log_step_fns = {'lr-schedule': lr_schedule}
        opt = optimizer(lr_schedule)
        model, (model_parameters, ema_parameters, state) = finetuning(
            fsb,
            model,
            model_parameters,
            ema_parameters,
            ema_rates,
            opt,
            pi_0,
            pi_1,
            batch_size,
            micro_batch_size,
            eps=eps,
            log_interval=log_interval//2,
            checkpoint_interval=checkpoint_interval//10,
            loss_fn=loss_fn,
            dtype=dtype,
            rngs=rngs,
            progbar=progbar,
            seed_time=seed_time,
            seed_noise=seed_noise,
            finetuning_steps=finetuning_steps,
            log_dir=log_dir,
            logger=logger,
            state=state,
            log_step_fns=log_step_fns,
            mp_policy=mp_policy,
            loss_scaler=loss_scaler,
            ckpt_finetuning=ckpt_finetuning
        )
    return model, (model_parameters, ema_parameters, state)
