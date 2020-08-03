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
from scipy.stats import spearmanr
from matplotlib.ticker import FormatStrFormatter


folder = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(folder, '..', 'summaries')
majorFormatter = FormatStrFormatter('%0.1f')


def load_summaries(d, metrics, merger, space):
    # Load json files
    for task in d:
        for r in range(len(d[task])):
            f = open(os.path.join(folder, d[task][r], 'sweep_summary.json'), 'rb')
            d[task][r] = json.load(f)
    # Find common architectures
    archs = []
    for task in d:
        for r in range(len(d[task])):
            archs.append(set([]))
            summ = d[task][r]['job_summaries']
            for i in range(len(summ)):
                arch = get_arch(summ[i], space)
                archs[-1].add(arch)
    common_archs = archs[0]
    for i in range(len(archs)):
        common_archs = common_archs & archs[i]
    # Create summs
    summs = {}
    for task in d:
        summs[task] = {}
        for r in range(len(d[task])):
            summ = d[task][r]['job_summaries']
            for i in range(len(summ)):
                arch = get_arch(summ[i], space)
                if arch not in common_archs:
                    continue
                if arch not in summs[task]:
                    summs[task][arch] = {}
                    for metric in metrics:
                        summs[task][arch][metric] = [scalar(summ[i][metric])]
                else:
                    for metric in metrics:
                        summs[task][arch][metric].append(scalar(summ[i][metric]))
    # Merge multiple runs into one number
    for task in d:
        for arch in common_archs:
            for metric in metrics:
                summs[task][arch][metric] = merger(summs[task][arch][metric])
    return summs


def get_arch(summ, space):
    if space == 'nas_bench':
        return str(summ['job_id'])
    elif space == 'darts':
        g = summ['net']['genotype']
        return str(g['normal'] + g['reduce'] + g['normal_concat'] + g['reduce_concat'])
    else:
        raise NotImplemented


def scalar(x):
    return x[-1] if isinstance(x, list) else x


def get_acc(space, dataset):
    acc = lambda x: 100 - x
    tasks = ['cls', 'rot', 'col', 'jig']
    # Determine which summaries to load
    d = defaultdict(list)
    if dataset == 'cifar10':
        for task in tasks:
            d[task] = [os.path.join(space, dataset, task, x) for x in ['1', '2', '3']]
    elif dataset == 'imagenet':
        for task in tasks:
            d[task] = [os.path.join(space, dataset, task, '1')]
    elif dataset == 'across':
        for task in tasks:
            d[task] = [os.path.join(space, 'cifar10', task, x) for x in ['1', '2', '3']]        
        d['imagenet_cls'] = [os.path.join(space, 'imagenet', 'cls', '1')]
    # Load summaries
    summs = load_summaries(d, ['min_test_top1'], np.mean, space)
    print('{} common archs for {} {}'.format(len(summs['cls']), space, dataset))
    # Postprocess
    for task in d:
        d[task] = []
    for arch in summs['cls']:
        for task in d:
            d[task].append(acc(summs[task][arch]['min_test_top1']))
    return d


def plot_darts_cifar10(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['cls']
    y = [d['rot'], d['col'], d['jig']]

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9*3, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y[0], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[0])[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(90.5,94)
    ax1.set_ylim(92,95)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("C10 rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y[1], robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[1])[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(90.5,94)
    ax2.set_ylim(30.3,30.8)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("C10 colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y[2], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[2])[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(90.5,94)
    ax3.set_ylim(70,79)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("C10 jigsaw puzzle task acc.", fontsize=27)

    f.text(0.5, -0.01, 'CIFAR-10 supervised classification accuracy (DARTS search space)', ha='center', fontsize=30)
    f.savefig('figs/fig2_top.pdf', bbox_inches = "tight")


def plot_bench_cifar10(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['cls']
    y = [d['rot'], d['col'], d['jig']]

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9*3, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y[0], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[0])[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(89,96)
    ax1.set_ylim(90,97)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("C10 rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y[1], robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[1])[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(89,96)
    ax2.set_ylim(30.4,30.8)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("C10 colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y[2], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[2])[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(89,96)
    ax3.set_ylim(69,82)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("C10 jigsaw puzzle task acc.", fontsize=27)

    f.text(0.5, -0.01, 'CIFAR-10 supervised classification accuracy (NAS-Bench-101 search space)', ha='center', fontsize=30)
    f.savefig('figs/fig2_bottom.pdf', bbox_inches = "tight")


def plot_darts_imagenet(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['cls']
    y = [d['rot'], d['col'], d['jig']]

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9*3, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y[0], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[0])[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(34,48)
    ax1.set_ylim(80,87)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("IN rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y[1], robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[1])[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(34, 48)
    ax2.set_ylim(32.25,32.55)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("IN colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y[2], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[2])[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(34, 49)
    ax3.set_ylim(40,63.5)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("IN jigsaw puzzle task acc.", fontsize=27)

    f.text(0.5, -0.01, 'ImageNet supervised classification accuracy (DARTS search space)', ha='center', fontsize=30)
    f.savefig('figs/fig3_top.pdf', bbox_inches = "tight")


def plot_bench_imagenet(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['cls']
    y = [d['rot'], d['col'], d['jig']]

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9*3, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y[0], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[0])[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(32,68)
    ax1.set_ylim(81.5,93)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("IN rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y[1], robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[1])[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(32, 68)
    ax2.set_ylim(32.3,33.6)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("IN colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y[2], robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y[2])[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(32, 68)
    ax3.set_ylim(58,76)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("IN jigsaw puzzle task acc.", fontsize=27)

    f.text(0.5, -0.01, 'ImageNet supervised classification accuracy (NAS-Bench-101 search space)', ha='center', fontsize=30)
    f.savefig('figs/fig3_bottom.pdf', bbox_inches = "tight")


def plot_darts_across(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['imagenet_cls']
    y1 = d['rot']
    y2 = d['col']
    y3 = d['jig']
    y4 = d['cls']

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7*4, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y1, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y1)[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(30,48)
    ax1.set_ylim(92.5,95)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("C10 rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y2, robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y2)[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(30, 48)
    ax2.set_ylim(30.4,30.8)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("C10 colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y3, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y3)[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(30, 48)
    ax3.set_ylim(72,78)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("C10 jigsaw puzzle task acc.", fontsize=27)

    sns.regplot(ax=ax4, color = palette[3], x=x, y=y4, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('C10 Supv.', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y4)[0])))
    ax4.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax4.set_xlim(30,48)
    ax4.set_ylim(91,94.5)
    ax4.xaxis.set_major_formatter(majorFormatter)
    ax4.yaxis.set_major_formatter(majorFormatter)
    ax4.set_ylabel("C10 supervised cls. acc.", fontsize=30)

    f.text(0.5, -0.01, 'ImageNet supervised classification accuracy (DARTS search space)', ha='center', fontsize=30)
    f.savefig('figs/fig4_top.pdf', bbox_inches = "tight")


def plot_bench_across(d):
    palette = sns.color_palette("Set2")
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 24

    x = d['imagenet_cls']
    y1 = d['rot']
    y2 = d['col']
    y3 = d['jig']
    y4 = d['cls']

    robust = True
    sns.set()
    sns.set(font_scale = 2)
    sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 50, 'axes.edgecolor':'0.15'})

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7*4, 6*1), tight_layout=True)

    sns.regplot(ax=ax1, color = palette[0], x=x, y=y1, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Rotation', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y1)[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax1.set_xlim(32,68)
    ax1.set_ylim(90,97)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    ax1.set_ylabel("C10 rotation task acc.", fontsize=30)

    sns.regplot(ax=ax2, color=palette[1], x=x, y=y2, robust=robust, scatter_kws={'alpha':0.5}, label='{:<1}\n{:>8}'.format('Colorization', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y2)[0])))
    ax2.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax2.set_xlim(32, 68)
    ax2.set_ylim(30.4,30.8)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.set_ylabel("C10 colorization task acc.", fontsize=28)

    sns.regplot(ax=ax3, color=palette[2], x=x, y=y3, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('Jigsaw', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y3)[0])))
    ax3.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax3.set_xlim(32, 68)
    ax3.set_ylim(69,82)
    ax3.xaxis.set_major_formatter(majorFormatter)
    ax3.yaxis.set_major_formatter(majorFormatter)
    ax3.set_ylabel("C10 jigsaw puzzle task acc.", fontsize=27)

    sns.regplot(ax=ax4, color = palette[3], x=x, y=y4, robust=robust, scatter_kws={'alpha':0.5}, label='{:<2}\n{:>8}'.format('C10 Supv.', r'$\rho$'+ '={:.2f}'.format(spearmanr(x, y4)[0])))
    ax4.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=-0.3, borderpad=0.1, markerscale=3, prop={'weight':'medium', 'size':'28'})
    ax4.set_xlim(32,68)
    ax4.set_ylim(89,97)
    ax4.xaxis.set_major_formatter(majorFormatter)
    ax4.yaxis.set_major_formatter(majorFormatter)
    ax4.set_ylabel("C10 supervised cls. acc.", fontsize=30)

    f.text(0.5, -0.01, 'ImageNet supervised classification accuracy (NAS-Bench-101 search space)', ha='center', fontsize=30)
    f.savefig('figs/fig4_bottom.pdf', bbox_inches = "tight")


if __name__ == '__main__':
    # Figure 2
    darts_cifar10 = get_acc('darts', 'cifar10')
    plot_darts_cifar10(darts_cifar10)
    bench_cifar10 = get_acc('nas_bench', 'cifar10')
    plot_bench_cifar10(bench_cifar10)
    # Figure 3
    darts_imagenet = get_acc('darts', 'imagenet')
    plot_darts_imagenet(darts_imagenet)
    bench_imagenet = get_acc('nas_bench', 'imagenet')
    plot_bench_imagenet(bench_imagenet)
    # Figure 4
    darts_across = get_acc('darts', 'across')
    plot_darts_across(darts_across)
    bench_across = get_acc('nas_bench', 'across')
    plot_bench_across(bench_across)
