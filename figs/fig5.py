# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from figs.fig234 import get_arch


folder = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(folder, '..', 'summaries')
_COLORS = sns.color_palette('Set2')
tasks = ['rot', 'col', 'jig', 'cls']


def parse(fnames, space):
    d = defaultdict(list)
    # Load json files
    archs = []
    for fname in fnames:
        f = open(os.path.join(folder, fname, 'sweep_summary.json'), 'rb')
        summs = json.load(f)['job_summaries']
        archs.append(set([get_arch(summ, space) for summ in summs]))
    # Find common architectures
    common_archs = archs[0]
    for i in range(len(archs)):
        common_archs = common_archs & archs[i]
    # Collect multiple runs
    for fname in fnames:
        f = open(os.path.join(folder, fname, 'sweep_summary.json'), 'rb')
        summs = json.load(f)['job_summaries']
        for summ in summs:
            arch = get_arch(summ, space)
            if arch not in common_archs:
                continue
            d[arch].append(100 - summ['min_test_top1'])
    # Merge multiple runs
    for arch in d:
        d[arch] = np.mean(d[arch])
    return d


def plot(d, space):
    r, c = 1, 1
    w, h = 4.4, 3.3
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(c * w, r * h))
    ax = axes

    n = 500
    labels = {
        'rot': 'C10 Rot',
        'col': 'C10 Color',
        'jig': 'C10 Jigsaw',
        'cls': 'C10 Supv.Cls',
    }
    colors = {
        'rot': _COLORS[0],
        'col': _COLORS[1],
        'jig': _COLORS[2],
        'cls': _COLORS[3],
    }
    for task in tasks:
        np.random.seed(2)
        m = 1
        mus, stds = [], []
        while m < n:
            s = []
            for i in range(int(np.ceil(n / m))):
                archs = np.random.choice(sorted(d[task].keys()), m, replace=False)
                best_arch, best_perf = None, 0
                for arch in archs:
                    if d[task][arch] > best_perf:
                        best_perf = d[task][arch]
                        best_arch = arch
                s.append(d['ref'][best_arch])
            mus.append(np.mean(s))
            stds.append(np.std(s))
            m *= 2
        mus, stds = np.array(mus), np.array(stds)
        ax.scatter(
            range(len(mus)), mus,
            color=colors[task], alpha=0.8, label=labels[task]
        )
        ax.fill_between(
            range(len(mus)), mus - 2 * stds, mus + 2 * stds,
            color=colors[task], alpha=0.05
        )
    if space == 'nas_bench':
        ax.set_title('NAS-Bench-101 search space', fontsize=16)
    elif space == 'darts':
        ax.set_title('DARTS search space', fontsize=16)
    ax.set_ylabel('IN1K Supv.Cls accuracy', fontsize=16)
    ax.grid(alpha=0.4)
    ax.set_xlabel('experiment size (log2)', fontsize=16)
    ax.set_xlim([-0.5, -0.5 + len(mus)])
    ax.legend(loc='lower right', prop={'size': 13})

    plt.tight_layout()
    plt.savefig('figs/{}_randexp.pdf'.format(space), dpi=300)


def random_experiment(space):
    # Determine which summaries to load
    d = {}
    d['ref'] = [os.path.join(space, 'imagenet', 'cls', '1')]
    for task in tasks:
        d[task] = [os.path.join(space, 'cifar10', task, x) for x in ['1', '2', '3']]
    # Load summaries
    for task in d:
        d[task] = parse(d[task], space)
    # Postprocess
    for task in tasks:
        archs = list(d[task].keys())
        for arch in archs:
            if arch not in d['ref']:
                del d[task][arch]
    plot(d, space)


if __name__ == '__main__':
    random_experiment('darts')
    random_experiment('nas_bench')
