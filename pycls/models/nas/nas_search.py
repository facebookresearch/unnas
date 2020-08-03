#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycls.models.common import Preprocess
from pycls.models.common import Classifier
from pycls.models.nas.operations import *
from pycls.models.nas.genotypes import DARTS_OPS as PRIMITIVES
from pycls.models.nas.genotypes import Genotype
from pycls.core.config import cfg
import pycls.core.logging as logging

logger = logging.get_logger(__name__)


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        if 'cifar' in cfg.TRAIN.DATASET:
            logger.info('Using CIFAR10 stem')
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            reduction_prev = False
        else:
            logger.info('Using ImageNet stem')
            C_curr = C
            self.stem0 = nn.Sequential(
                nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, C // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            reduction_prev = True

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_layers = [layers//3] if cfg.TASK == 'seg' else [layers//3, 2*layers//3]
        for i in range(layers):
            if i in reduction_layers:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.classifier = Classifier(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        input = Preprocess(input)
        if 'cifar' in cfg.TRAIN.DATASET:
            s0 = s1 = self.stem(input)
        else:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        logits = self.classifier(s1, input.shape[2:])
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


class NAS_Search(nn.Module):
    """NAS net wrapper (delegates to nets from DARTS)."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet', 'imagenet22k', 'cityscapes'], \
            'Training on {} is not supported'.format(cfg.TRAIN.DATASET)
        super(NAS_Search, self).__init__()
        logger.info('Constructing NAS_Search: {}'.format(cfg.NAS))
        if cfg.TASK == 'seg':
            criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
        self.net_ = Network(
            C=cfg.NAS.WIDTH,
            num_classes=cfg.MODEL.NUM_CLASSES,
            layers=cfg.NAS.DEPTH,
            criterion=criterion
        )

    def _loss(self, input, target):
        return self.net_._loss(input, target)

    def arch_parameters(self):
        return self.net_.arch_parameters()

    def genotype(self):
        return self.net_.genotype()

    def forward(self, x):
        return self.net_.forward(x)
