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
from tqdm import tqdm
import pickle
from einops import repeat
from .loss_functions import mse, step
from .ema import update_ema
from .. import distributed as dist
from ..sampling_lib import em, pf
from .. import mixed_precision as mp
from .. import data
from flax import serialization
from flax.training import checkpoints
from flax.serialization import from_state_dict


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
                    stage='pretraining', target=None):
    ckpt = checkpoints.restore_checkpoint(ckpt_pth, target=target)
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
      step=ckpt['state'].step,
      apply_fn=state.apply_fn,
      params=jax.device_put(ckpt['state'].params, dist.REPLICATE_SHARDING),
      tx=state.tx,
      opt_state=jax.device_put(opt_state, dist.REPLICATE_SHARDING),
    )
    ema_parameters = [jax.device_put(e, dist.REPLICATE_SHARDING) for e in ckpt['ema_parameters'] ] # [ckpt['ema_parameters'][k]
    #ema_parameters = ckpt['ema_parameters']# [ckpt['ema_parameters'][k]
                     #  for k in sorted(ckpt['ema_parameters'].keys())]
    ema_rates = ckpt['ema_rates'] # [ckpt['ema_rates'][k] for k in sorted(ckpt['ema_rates'].keys())]
    key_noise_fwd = ckpt['key_noise_fwd']
    key_noise_bwd = ckpt['key_noise_bwd']
    key_time = ckpt['key_time']
    key_dropout = ckpt['key_dropout']
    (key_noise_fwd,
     key_noise_bwd,
     key_time,
     key_dropout
    ) = dist.map_sharding(dist.REPLICATE_SHARDING, key_noise_fwd,
     key_noise_bwd,
     key_time,
     key_dropout)
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


def log_step(logger, stp, apply_fn, params, X_0, X_1, directions, mean_loss,
             stage=None, solver=em, solver_steps=100, pi_0=None, pi_1=None,
             karras_pred=False, **log_step_fns):
    # starting points
    X = jnp.concatenate([X_0, X_1], axis=0)
    X = jax.device_put(X, dist.DP_SHARDING)
    # endpoints
    imgs_forwd, imgs_backwd = jnp.split(
        solver(apply_fn, params, X, directions, num_steps=solver_steps,
               karras_pred=karras_pred),
        2
    )
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


def time_sampling(key, n, dtype=jnp.float32, t_offset=0.01):
    r"""
    Args:
        key: PRNGKey for sampling
        n: number of samples
        dtype: (optional) the data-type of time values (e.g. float32, bfloat16)
        t_offset: (optional) a small number, scaling the uniform distribution to
                  [t_offset, 1-t_offset]. Useful for numeric stability in the
                  loss.
    Returns:
        samples in range [t_offset, 1-t_offset], with shape (n, 1)
    """
    return uniform(key, (n, 1), dtype=dtype, minval=t_offset, maxval=1-t_offset)


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


def write(model_parameters, ema_parameters, ema_rates, log_dir, step,
          verbose=True, pfx=''):
    r"""
    simple chekpointing method, all checkpoints are pickle-files

    Args:
        model_parameters: parameters of online model
        ema_parameters: list of EMA parameters
        ema_rates: list of respective EMA rates
        log_dir: directory where checkpoints will be saved
        step: optimization step to indicate in filename
        verbose: a flag, if set a notification will be printed
    """
    # step-identifier
    descriptor = f'_{pfx + "_" if pfx != "" else ""}step_{step}.pkl'
    # model parameters
    fname = os.path.join(log_dir, f'model{descriptor}')
    with open(fname, 'wb') as f:
        pickle.dump(model_parameters, f)
    if verbose:
        print(f'wrote {fname}', flush=True)
    # EMA parameters
    for ema_params, ema_rate in zip(ema_parameters, ema_rates):
        fname = os.path.join(log_dir, f'ema_{ema_rate}{descriptor}')
        with open(fname, 'wb') as f:
            pickle.dump(ema_params, f)
        if verbose:
            print(f'wrote {fname}', flush=True)


@partial(jax.jit, static_argnames=['eps'])
def reciprocal_interpolation(X_0, X_1, Z, t, eps=1):
    r"""
    This is simply flow matching with some added entropy
    """
    return cast2d(1-t) * X_0 \
            + cast2d(t) * X_1 \
            + cast2d(jnp.sqrt(eps * (1-t) * t)) * Z


def pretraining(
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
        karras_pred=False,
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
    # sanity checks
    assert_sanity(batch_size, micro_batch_size)

    # scale pretraining steps to a micro-batch perspective
    acc_steps = resolve_acc_steps(batch_size, micro_batch_size)
    pretraining_steps = pretraining_steps * acc_steps

    # seeds for time- and noise samplers
    key_time = jax.random.PRNGKey(seed_time)
    key_noise = jax.random.PRNGKey(seed_noise)

    # flax training state initialization
    state = TrainState.create(apply_fn=model.apply, params=model_parameters,
                              tx=optimizer)

    # half micro batch size for splitting batches w.r.t. fwd & bwd SDEs
    half_mb = micro_batch_size // 2
    # direction-indices for fwd & bwd SDEs
    # NOTE: for simplicity, the first half of every batch corresponds to the fwd
    #       path and vice versa
    _1 = jnp.ones((half_mb, 1), dtype=jnp.int32)   # 1 <- fwd
    _0 = jnp.zeros((half_mb, 1), dtype=jnp.int32)  # 0 <- bwd
    directions = jnp.concatenate([_1, _0], axis=0)
    directions = jax.device_put(directions, dist.DP_SHARDING)

    # direct reference for dropout PRNG
    key_dropout = rngs['dropout']

    # load from checkpoint
    keys = dict(
        key_noise_fwd=key_noise,
        key_noise_bwd=key_noise,
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
    if ckpt_pretraining is not None:
        ckpt_pretraining = os.path.abspath(ckpt_pretraining)
        (state,
         ema_parameters, ema_rates,
         key_noise, key_noise_bwd,
         key_time, key_dropout) = load_checkpoint(ckpt_pretraining,
                                                  state,
                                                  stage='pretraining',
                                                  target=ckpt)
        it_start = state.step
    else:
        it_start = 0

    # skipping training, if no steps specified
    if pretraining_steps is None or pretraining_steps <= 0:
        return model, state.params, ema_parameters, state

    # logging setup & running mean loss
    mean_loss = 0
    if progbar:
        iterator = tqdm(jnp.arange(pretraining_steps))
    else:
        iterator = range(pretraining_steps)

    # Pretraining loop
    for iteration in iterator:
        # new PRNG states
        key_dropout, subkey_dropout = jax.random.split(key_dropout)
        key_time, subkey_time = jax.random.split(key_time)
        key_noise, subkey_noise = jax.random.split(key_noise)
        # distributed PRNG states
        dropout_rngs = jax.device_put(subkey_dropout, dist.REPLICATE_SHARDING)

        # (I) Sampling from data
        # sample from terminal distributions
        X_0, _ = pi_0.sample(micro_batch_size)
        X_1, _ = pi_1.sample(micro_batch_size)
        X_0, X_1 = dist.map_sharding(dist.DP_SHARDING, X_0, X_1)

        # (II) Sampling from marginals of paths
        # sample time
        t = time_sampling(subkey_time, micro_batch_size, dtype=dtype)
        # sample noise
        Z = noise_sampling(subkey_noise, X_0.shape, dtype=dtype)
        # sharding (for distributed training setup, see jax_lib.distributed)
        t, Z, = dist.map_sharding(dist.DP_SHARDING, t, Z)
        # sample from path (reciprocal interpolation between terminals)
        X_t = reciprocal_interpolation(X_0, X_1, Z, t, eps)
        # time directions
        t_input = directions * t + (1-directions) * (1-t)
        # assign targets to respective forw & back paths
        targets = jnp.concatenate(
            [(X_1[:half_mb] - X_t[:half_mb]) / cast2d(1-t[:half_mb]),  # fwd
             (X_0[half_mb:] - X_t[half_mb:]) / cast2d(t[half_mb:])],   # bwd
            axis=0
        )

        # (III) Optimization step
        # sharding (for distributed training setup, see jax_lib.distributed)
        X_t, targets, t_input = dist.map_sharding(dist.DP_SHARDING,
                                                  X_t, targets, t_input)
        # optimization step
        loss, state, grads, loss_scaler = step(
            state, X_t, targets, t_input, directions, dropout_rngs, eps=eps,
            mp_policy=mp_policy, loss_scaler=loss_scaler, karras_pred=karras_pred
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
            log_step(logger, int(stp), state.apply_fn, state.params,
                     X_0[:half_mb], X_1[:half_mb], directions, mean_loss,
                     stage='pretraining', pi_0=pi_0, pi_1=pi_1,
                     karras_pred=karras_pred, **log_step_fns)
            mean_loss = 0
        # saving checkpoints
        if stp % checkpoint_interval == 0:
            keys = dict(
                key_noise_fwd=key_noise,
                key_noise_bwd=key_noise,
                key_time=key_time,
                key_dropout=key_dropout
            )
            write_checkpoint(
                os.path.abspath(log_dir), int(stp), pfx='pretraining', keep_max=keep_max, verbose=True,
                state=state,
                opt_state=serialization.to_state_dict(state.opt_state),
                ema_parameters=ema_parameters,
                ema_rates=ema_rates,
                **keys
            )
    return model, state.params, ema_parameters, state


def finetuning(
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
        solver_steps=100,
        solver=em,
        mp_policy=mp.DEFAULT_POLICY,
        loss_scaler=mp.DEFAULT_LOSS_SCALER,
        karras_pred=False,
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
    assert_sanity(batch_size, micro_batch_size, stage='finetuning')

    # scale pretraining steps to a micro-batch perspective
    acc_steps = resolve_acc_steps(batch_size, micro_batch_size)
    finetuning_steps = finetuning_steps * acc_steps

    # seeds for time- and noise samplers
    key_time = jax.random.PRNGKey(seed_time)
    key_noise = jax.random.PRNGKey(seed_noise)
    key_sde_fwd = jax.random.PRNGKey(seed_noise+1)
    key_sde_bwd = jax.random.PRNGKey(seed_noise+2)

    # flax training state initialization
    state = TrainState.create(apply_fn=model.apply, params=model_parameters,
                              tx=optimizer)

    # half micro batch size for splitting batches w.r.t. fwd & bwd SDEs
    half_mb = micro_batch_size // 2
    # direction-indices for fwd & bwd SDEs
    _1 = jnp.ones((half_mb, 1), dtype=jnp.int32)     # 1 <- fwd
    _0 = jnp.zeros((half_mb, 1), dtype=jnp.int32)    # 0 <- bwd
    directions = jnp.concatenate([_1, _0], axis=0)
    _1, _0, directions = dist.map_sharding(dist.DP_SHARDING, _1, _0, directions)

    # logging setup & running mean loss
    mean_loss = 0
    if progbar:
        iterator = tqdm(jnp.arange(finetuning_steps))
    else:
        iterator = range(finetuning_steps)

    # direct reference for dropout PRNG
    key_dropout = rngs['dropout']

    # Finetuning loop
    for iteration in iterator:
        # new PRNG states
        key_dropout, subkey_dropout = jax.random.split(key_dropout)
        key_time, subkey_time = jax.random.split(key_time)
        key_noise, subkey_noise = jax.random.split(key_noise)
        key_sde_fwd, subkey_sde_fwd = jax.random.split(key_sde_fwd)
        key_sde_bwd, subkey_sde_bwd = jax.random.split(key_sde_bwd)
        # distributed PRNG states
        dropout_rngs = jax.device_put(subkey_dropout, dist.REPLICATE_SHARDING)

        # (I) Sampling from data
        # sample from terminal distributions
        X_0, _ = pi_0.sample(half_mb)
        X_1, _ = pi_1.sample(half_mb)
        X_0, X_1 = dist.map_sharding(dist.DP_SHARDING, X_0, X_1)
        # (a) solving fwd
        sde_X_1 = solver(state.apply_fn, state.params, X_0, _1, key=subkey_sde_fwd,
                         num_steps=solver_steps, karras_pred=karras_pred)
        # (b) solving bwd
        sde_X_0 = solver(state.apply_fn, state.params, X_1, _0, key=subkey_sde_bwd,
                         num_steps=solver_steps, karras_pred=karras_pred)

        # (II) Sampling from marginals of paths
        # sample time for forward & backward processes
        t_fwd, t_bwd = jnp.split(
            time_sampling(subkey_time, 2*half_mb, dtype=dtype), 2
        )
        # sample noise for forward & backward processes
        Z_fwd, Z_bwd = jnp.split(
            noise_sampling(subkey_noise, (2*half_mb, *X_0.shape[1:]),
                           dtype=dtype),
            2
        )
        # sharding (for distributed training setup, see jax_lib.distributed)
        t_fwd, t_bwd, Z_fwd, Z_bwd, sde_X_0, sde_X_1 = dist.map_sharding(
            dist.DP_SHARDING, t_fwd, t_bwd, Z_fwd, Z_bwd, sde_X_0, sde_X_1
        )
        # sample from path (reciprocal interpolation between terminals)
        X_t_fwd = reciprocal_interpolation(sde_X_0, X_1, Z_fwd, t_fwd, eps)
        X_t_bwd = reciprocal_interpolation(X_0, sde_X_1, Z_bwd, t_bwd, eps)

        # (III) Optimization step
        # sharding (for distributed training setup, see jax_lib.distributed)
        X_t = jnp.concatenate([X_t_fwd, X_t_bwd], axis=0)
        t = jnp.concatenate([t_fwd, 1-t_bwd], axis=0)
        targets = jnp.concatenate(
            [(X_1 - X_t_fwd) / cast2d(1-t_fwd),   # fwd
             (X_0 - X_t_bwd) / cast2d(t_bwd)],    # bwd
            axis=0
        )
        targets, X_t, t = dist.map_sharding(dist.DP_SHARDING, targets, X_t, t)
        loss, state, grads, loss_scaler = step(
            state, X_t, targets, t, directions, dropout_rngs, eps=eps,
            mp_policy=mp_policy, loss_scaler=loss_scaler, karras_pred=karras_pred
        )
        # update ema parameters every step
        if (iteration+1) % float(acc_steps) == 0:
            ema_parameters = [update_ema(ep, state.params, er)
                              for ep, er in zip(ema_parameters, ema_rates)]

        # (IV) Logging & saving weights
        mean_loss += loss / (log_interval * acc_steps)
        stp = (iteration+1) / acc_steps
        # logging progress
        if stp % log_interval == 0 and logger is not None:
            log_step(logger, int(stp), state.apply_fn, state.params, X_0, X_1,
                     directions, mean_loss, stage='finetuning', pi_0=pi_0,
                     pi_1=pi_1, karras_pred=karras_pred, **log_step_fns)
            mean_loss = 0
        # saving checkpoints
        if stp % checkpoint_interval == 0:
            keys = dict(
                key_noise_fwd=key_noise,
                key_noise_bwd=key_noise,
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
    return model, state.params, ema_parameters, state


def train(
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
        karras_pred=False,
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
    lr_schedule = lr_sched(alpha, max(pretraining_steps, 10001))
    log_step_fns = {'lr-schedule': lr_schedule,
                    'loss-scale': lambda _ : loss_scaler.loss_scale}
    opt = optimizer(lr_schedule)
    model, model_parameters, ema_parameters, state = pretraining(
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
        karras_pred=karras_pred,
        ckpt_pretraining=ckpt_pretraining
    )
    # using half the learning rate for fine-tuning
    if finetuning_steps > 0:
        lr_schedule = lr_sched(alpha/2, finetuning_steps, warmup_steps=1000)
        log_step_fns = {'lr-schedule': lr_schedule}
        opt = optimizer(lr_schedule)
        model, model_parameters, ema_parameters, state = finetuning(
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
            finetuning_steps=finetuning_steps,
            log_dir=log_dir,
            logger=logger,
            state=state,
            log_step_fns=log_step_fns,
            mp_policy=mp_policy,
            loss_scaler=loss_scaler,
            karras_pred=karras_pred
        )
    return model, model_parameters, ema_parameters, state
