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
import sys
import os
import jax
import optax
import argparse
import yaml
import pickle
from functools import partial
from .. import data
from .. import model_zoo
from .. import log_lib
from .. import distributed
from .. import mixed_precision
from ..MAfBM import fSB
from ..MAfBM.alpha_diffusion_MAfBM import pretraining, finetuning, \
                                          train, resolve_acc_steps
import numpy as np


jax.config.update('jax_threefry_partitionable', True)


# default RNG key for model initialization
DEFAULT_KEY = 0


def model_constructor(name):
    return vars(model_zoo)[name]


def dset_constructor(name):
    return vars(data)[name]


def opt_constructor(name):
    return vars(optax)[name]


def precision(name):
    return vars(jnp)[name]


def logger_type(name):
    return vars(log_lib)[name]


def str2bool(v):
    return v.lower() in ['true', 't', 'yes', 'y', '1']


def args_parser():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--pi0', type=dset_constructor, default=data.MNIST)
    ap.add_argument('--pi1', type=dset_constructor, default=data.EMNIST)
    ap.add_argument('--root0', type=str, default='datasets')
    ap.add_argument('--root1', type=str, default='datasets')
    ap.add_argument('--size', type=int, default=32)
    # architecture
    ap.add_argument('--arch', type=model_constructor,
                    default=model_zoo.DiT_S_4)
    ap.add_argument('--precision', type=precision, default=jnp.float32)
    ap.add_argument('--karras_pred', type=str2bool, default=False)
    # weights
    ap.add_argument('--model_params', type=str, default=None)
    ap.add_argument('--ema_params', nargs='+', type=str, default=None)
    # MAfBM
    ap.add_argument('--H', type=float, default=0.5)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--MAfBM_norm', type=str2bool, default=True)
    # optimizer
    ap.add_argument('--optimizer', type=opt_constructor, default=optax.lion)
    ap.add_argument('--lr', type=float, default=0.0002)
    ap.add_argument('--sqrt_eps', type=float, default=1.0)
    ap.add_argument('--ema_rates', nargs='+', type=float, default=[0.999])
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--micro_batch_size', type=int, default=128)
    # trianing loop
    ap.add_argument('--pretraining_steps', type=int, default=int(1e5))
    ap.add_argument('--finetuning_steps', type=int, default=0)
    # I/O
    ap.add_argument('--out_dir', type=str, default='outputs')
    ap.add_argument('--log_dir', type=str, default='logs')
    ap.add_argument('--pth_in', type=str, default='')
    # logger
    ap.add_argument('--log_interval', type=int, default=int(1e4))
    ap.add_argument('--logger', type=logger_type, default=log_lib.WandBLogger)
    ap.add_argument('--wandb_key', type=str, default='')
    ap.add_argument('--wandb_entity', type=str, default='diffhhi')
    ap.add_argument('--wandb_group', type=str, default=None)
    ap.add_argument('--progbar', action='store_true')
    # model checkpointing
    ap.add_argument('--checkpoint_interval', type=int, default=int(5e4))
    ap.add_argument('--ckpt_pretraining', type=str, default=None)
    ap.add_argument('--ckpt_finetuning', type=str, default=None)
    # cluster specific arguments
    ap.add_argument('--job_id', type=str, default='')
    return ap


def load_params(*paths):
    def _load(p):
        with open(p, 'rb') as f:
            params = pickle.load(f)
        return params
    return [_load(p) for p in paths]


def get_lr_sched(lr, steps_total, warmup_steps=None):
    # 10K warmup steps to accomodate larger models
    steps_warmup = warmup_steps if warmup_steps is not None else 10000
    steps_decay = steps_total # NOTE: - steps_warmup is applied by optax
    return optax.schedules.warmup_cosine_decay_schedule(
        init_value=lr/1000,
        peak_value=lr,
        warmup_steps=steps_warmup,
        decay_steps=steps_decay,
        end_value=lr/100,
        exponent=1.0
    )


def init_opt(opt_fn, lr, multisteps=1):
    opt = optax.chain(
        # unsure if clipping compromises lion
        # optax.clip_by_global_norm(1.0),
        opt_fn(lr, weight_decay=1e-4)
    )
    # gradient accumulation, if specified
    opt = optax.MultiSteps(opt, every_k_schedule=multisteps)
    return opt


def io_sanity_checks(**kwargs):
    for key in kwargs.keys():
        if key.endswith('dir'):
            if not os.path.isdir(kwargs[key]):
                os.makedirs(kwargs[key])


def init_model(model, key, *args):
    model_parameters = jax.device_put(model.init(key, *args),
                                      distributed.REPLICATE_SHARDING)
    return model_parameters


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
      step=ckpt['state'].step, #['step'],
      apply_fn=state.apply_fn,
      params=ckpt['state'].params, #['params'],
      tx=state.tx,
      opt_state=opt_state
    )
    ema_parameters =ckpt['ema_parameters'] #[ckpt['ema_parameters'][k]
                     # for k in sorted(ckpt['ema_parameters'].keys())]
    ema_rates = ckpt['ema_rates']#[ckpt['ema_rates'][k] for k in sorted(ckpt['ema_rates'].keys())]
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


def eval_sampling(fsb, apply_fn, params, pi_0, pi_1, solver_steps=100,
                  num_reps=10, batch_size=256):
    Y_0 = 0
    # num_samples
    N_0 = len(pi_0)
    N_1 = len(pi_1)
    # preparing outputs
    inputs_fwd = []
    inputs_bwd = []
    outputs_fwd = []
    outputs_bwd = []
    # directions
    _1 = jnp.ones(batch_size, dtype=jnp.int32)
    _0 = jnp.zeros(batch_size, dtype=jnp.int32)
    # RNG
    key = jax.random.PRNGKey(0)
    # sharding
    params = jax.device_put(params, dist.REPLICATE_SHARDING)
    key = jax.device_put(key, dist.REPLICATE_SHARDING)
    _0, _1 = dist.map_sharding(dist.DP_SHARDING, _0, _1)
    #
    for R in range(num_reps):
        print('rep', R)
        idx_0 = 0
        idx_1 = 0
        # backward
        print('bwd')
        pb = tqdm(total=N_1)
        while idx_1 < N_1:
            key, subkey = jax.random.split(key)
            # indices range
            end = min(N_1, idx_1 + batch_size)
            idcs = jnp.arange(idx_1, end)
            # sample
            X_1, _ = pi_1.sample(batch_size, idcs_X=idcs)
            X_1 = jax.device_put(X_1, dist.DP_SHARDING)
            zt_bwd = em_fwd(apply_fn, params, fsb, X_1, _0[:end-idx_1],
                            num_steps=solver_steps, Y_0=Y_0,
                            key=subkey)
            imgs_backwd = zt_bwd[..., 0]
            if not isinstance(pi_0, data.LatentDataset):
                imgs_backwd = pi_0.unscale(imgs_backwd)
                imgs_backwd = (imgs_backwd + 1) / 2
                imgs_backwd = jnp.clip(imgs_backwd, 0, 1)
                X_1 = pi_1.unscale(X_1)
                X_1 = (X_1 + 1) / 2
                X_1 = jnp.clip(X_1, 0, 1)
            # storing inputs & outputs
            inputs_bwd.append(X_1)
            outputs_bwd.append(imgs_backwd)
            # increment -> next batch
            idx_1 = end
            pb.update(batch_size)
        # forward
        print('fwd')
        pb = tqdm(total=N_0)
        while idx_0 < N_0:
            key, subkey = jax.random.split(key)
            # indices range
            end = min(N_0, idx_0 + batch_size)
            idcs = jnp.arange(idx_0, end)
            # sample
            X_0, _ = pi_0.sample(batch_size, idcs_X=idcs)
            X_0 = jax.device_put(X_0, dist.DP_SHARDING)
            zt_fwd = em_fwd(apply_fn, params, fsb, X_0, _1[:end-idx_0],
                            num_steps=solver_steps, Y_0=Y_0,
                            key=subkey)
            imgs_forwd = zt_fwd[..., 0]
            if not isinstance(pi_1, data.LatentDataset):
                imgs_forwd = pi_1.unscale(imgs_forwd)
                imgs_forwd = (imgs_forwd + 1) / 2
                imgs_forwd = jnp.clip(imgs_forwd, 0, 1)
                X_0 = pi_0.unscale(X_0)
                X_0 = (X_0 + 1) / 2
                X_0 = jnp.clip(X_0, 0, 1)
            # storing inputs & outputs
            inputs_fwd.append(X_0)
            outputs_fwd.append(imgs_forwd)
            # increment -> next batch
            idx_0 = end
            pb.update(batch_size)
    # concatenate results
    inputs_fwd, outputs_fwd, inputs_bwd, outputs_bwd = map(
        partial(jnp.concatenate, axis=0),
        (inputs_fwd, outputs_fwd, inputs_bwd, outputs_bwd)
    )
    return (inputs_fwd, outputs_fwd), (inputs_bwd, outputs_bwd)


if __name__ == '__main__':
    args = args_parser().parse_args()
    # I/O sanity
    io_sanity_checks(**vars(args))
    # logger
    is_wandb = (args.logger == log_lib.WandBLogger)
    if is_wandb:
        log_args = [args.log_dir, args.wandb_key, args.wandb_entity,
                    f'{args.arch.__name__} | JOB-ID:{args.job_id}']
        log_kwargs = dict(group=args.wandb_group)
    else:
        log_args = [args.log_dir]
        log_kwargs = {}
    logger = args.logger(*log_args, **log_kwargs)
    # data
    pi_0 = args.pi0(root=args.root0, train=False, size=args.size)
    pi_1 = args.pi1(root=args.root1, train=False, size=args.size)
    ref_pi_0 = args.pi0(root=args.root0, train=True, size=args.size)
    ref_pi_1 = args.pi1(root=args.root1, train=True, size=args.size)
    # dummy tensors for model init
    xshape, _ = pi_0.shape
    dummy_x = jnp.zeros([args.micro_batch_size, *xshape])
    dummy_y = jnp.zeros([args.micro_batch_size], dtype=jnp.int32)
    dummy_t = jnp.zeros([args.micro_batch_size])
    dummy_args = (dummy_x, dummy_t, dummy_y)
    # model
    model = args.arch(dtype=args.precision, param_dtype=jnp.float32)
    key_model = jax.random.PRNGKey(DEFAULT_KEY)
    if args.model_params is None:
        model_parameters = init_model(model, key_model, *dummy_args)
    else:
        model_parameters = load_params(args.model_params)
    # EMA models
    if args.ema_params is None:
        keys_ema = [jax.random.PRNGKey(DEFAULT_KEY)
                    for i in range(len(args.ema_rates))]
        ema_parameters = [init_model(model, key, *dummy_args)
                          for key in keys_ema]
    else:
        ema_parameters = load_params(*args.ema_params)
    # optimizer
    lr_sched = get_lr_sched
    acc_steps = resolve_acc_steps(args.batch_size, args.micro_batch_size)
    opt = partial(init_opt, args.optimizer, multisteps=acc_steps)
    # mixed precision policy & loss scaler
    policy = mixed_precision.policy(args.precision)
    loss_scaler = mixed_precision.loss_scaling(args.precision)
    # log size
    print(model_zoo.param_count(model_parameters, readable=True), flush=True)
    # log args
    with open(os.path.join(args.log_dir, 'args.yml'), 'w') as f:
        yaml.dump({k: str(v) for k, v in vars(args).items()}, f)
    with open(os.path.join(args.log_dir, 'sys_argv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    #
    lr_schedule = lr_sched(args.lr, 10001, warmup_steps=10000)
    optimizer = opt(lr_schedule)
    state = TrainState.create(apply_fn=model.apply, params=model_parameters,
                              tx=optimizer)
    # configure fSB params
    fsb = fSB(H=args.H, K=args.K, g_max=args.sqrt_eps, norm=args.MAfBM_norm)
    print(fsb)
    # eval
    # load checkpoint
    keys = dict(
       key_noise_fwd=jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING),
       key_noise_bwd=jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING),
       key_time=jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING),
       key_dropout=jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING)
    )
    ckpt = dict(
        state=state,
        opt_state=serialization.to_state_dict(state.opt_state),
        ema_parameters=ema_parameters,
        ema_rates=args.ema_rates,
        **keys
    )
    if args.ckpt_finetuning is not None:
        ckpt = dict(
                **ckpt,
        key_sde_fwd =jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING),
        key_sde_bwd =jax.device_put(jax.random.PRNGKey(0), dist.REPLICATE_SHARDING)
                )
        (
            state,
            ema_parameters, ema_rates,
            key_noise_fwd, key_noise_bwd,
            key_time, key_dropout,
            key_sde_fwd, key_sde_bwd
        ) = load_checkpoint(
            os.path.abspath(args.ckpt_finetuning), state, verbose=True,
            stage='finetuning', target=ckpt
        )
    else:
        ( state,
        ema_parameters, ema_rates,
        key_noise_fwd, key_noise_bwd,
        key_time, key_dropout ) = load_checkpoint(
            os.path.abspath(args.ckpt_pretraining), state, verbose=True,
            stage='pretraining', target=ckpt
        )
    # model selection
    apply_fn = state.apply_fn
    params = ema_parameters[0]
    # sample
    (inputs_fwd, outputs_fwd), (inputs_bwd, outputs_bwd) = eval_sampling(
        fsb=fsb,
        apply_fn=apply_fn,
        params=params,
        pi_0=pi_0,
        pi_1=pi_1,
        solver_steps=200,
        num_reps=10,
        batch_size=args.batch_size
    )
    # save samples
    np.save(os.path.join(args.out_dir, 'fwd_inputs'), np.array(inputs_fwd))
    np.save(os.path.join(args.out_dir, 'fwd_outputs'), np.array(outputs_fwd))
    np.save(os.path.join(args.out_dir, 'bwd_inputs'), np.array(inputs_bwd))
    np.save(os.path.join(args.out_dir, 'bwd_outputs'), np.array(outputs_bwd))
    # save reference
    true_dist_fwd, _ = ref_pi_1.sample(len(ref_pi_1), idcs_X=jnp.arange(len(ref_pi_1)))
    true_dist_bwd, _ = ref_pi_0.sample(len(ref_pi_0), idcs_X=jnp.arange(len(ref_pi_0)))
    if not isinstance(pi_0, data.LatentDataset):
        true_dist_bwd = ref_pi_0.unscale(true_dist_bwd)
        true_dist_bwd = (true_dist_bwd + 1) / 2
        true_dist_bwd = jnp.clip(true_dist_bwd, 0, 1)
    if not isinstance(pi_1, data.LatentDataset):
        true_dist_fwd = ref_pi_0.unscale(true_dist_fwd)
        true_dist_fwd = (true_dist_fwd + 1) / 2
        true_dist_fwd = jnp.clip(true_dist_fwd, 0, 1)
    np.save(os.path.join(args.out_dir, 'true_dist_fwd'), np.array(true_dist_fwd))
    np.save(os.path.join(args.out_dir, 'true_dist_bwd'), np.array(true_dist_bwd))
