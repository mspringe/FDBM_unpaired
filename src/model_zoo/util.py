"""
This module provides simple helpoer functions to display model parameters.

Author: Maximilian Springenberg
Affiliation: Fraunhofer HHI
License: This source code is licensed under the CC-by-NC license found in the
         LICENSE file in the root directory of this source tree.
"""
import jax.tree_util as jtu
from flax.linen import tabulate


def frame_text(txt):
    blue = '\033[94m'
    neutral = '\033[0m'
    cstrip = lambda s: s.replace(blue, '').replace(neutral, '')
    spacing = len(cstrip(txt))
    s = f'┌─{"─"*spacing}─┐\n' \
        f'│ {txt} │\n' \
        f'└─{"─"*spacing}─┘'
    return s


def human_readable(n, frame=False, pfx=''):
    blue = '\033[94m'
    neutral = '\033[0m'
    lbls = ['', 'K', 'M', 'B', 'T']
    i = 0
    while len(str(int(n))) > 3:
        n /= 1e3
        i += 1
        if i >= (len(lbls)-1):
            break
    s = f'{blue}{pfx}{neutral}{n:.1f}{lbls[i]}'
    return s if not frame else frame_text(s)


def param_count(params, readable=False):
    n_params = sum(p.size for p in jtu.tree_leaves(params))
    return n_params if not readable else human_readable(n_params, frame=True,
                                                        pfx='#params ')


def model_summary(model, rngs, *inputs, **kwargs):
    tabulate_fn = tabulate(model, rngs, **kwargs)
    return tabulate_fn(*inputs)
