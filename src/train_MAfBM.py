r"""
This module provides the training script; exemplary execution:


>>> python jax_lib/train.py --arch DiT_B_4 --precision bfloat16


Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import sys
import os
import jax
import jax.numpy as jnp
import optax
import argparse
import yaml
import pickle
from functools import partial
from . import data
from . import model_zoo
from . import log_lib
from . import distributed
from . import mixed_precision
from .MAfBM import fSB
from .MAfBM.alpha_diffusion_MAfBM import (
    pretraining, finetuning, train, resolve_acc_steps
)


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
    pi_0 = args.pi0(root=args.root0, train=True, size=args.size)
    pi_1 = args.pi1(root=args.root1, train=True, size=args.size)
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
    # configure fSB params
    fsb = fSB(H=args.H, K=args.K, g_max=args.sqrt_eps, norm=args.MAfBM_norm)
    print(fsb)
    # start training
    train(
        fsb,
        model, model_parameters, ema_parameters, args.ema_rates,
        pi_0=pi_0,
        pi_1=pi_1,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        alpha=args.lr,
        eps=args.sqrt_eps**2,
        log_interval=args.log_interval,
        dtype=args.precision,
        optimizer=opt,
        lr_sched=lr_sched,
        checkpoint_interval=args.checkpoint_interval,
        pretraining_steps=args.pretraining_steps,
        finetuning_steps=args.finetuning_steps,
        rngs={'dropout': jax.random.PRNGKey(1)},
        progbar=args.progbar,
        log_dir=args.out_dir,
        logger=logger,
        mp_policy=policy,
        loss_scaler=loss_scaler,
        ckpt_pretraining=args.ckpt_pretraining,
        ckpt_finetuning=args.ckpt_finetuning
    )
