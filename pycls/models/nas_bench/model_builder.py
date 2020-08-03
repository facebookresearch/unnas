# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

import pycls.core.logging as logging
import pycls.models.nas_bench.base_ops as base_ops
from pycls.models.nas_bench.model_spec import ModelSpec
from pycls.models.common import Preprocess
from pycls.models.common import Classifier
from pycls.core.config import cfg


logger = logging.get_logger(__name__)

class NAS_Bench(nn.Module):
    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet', 'imagenet22k'], \
            'Training on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10', 'imagenet', 'imagenet22k'], \
            'Testing on {} is not supported'.format(cfg.TEST.DATASET)
        super(NAS_Bench, self).__init__()
        logger.info('Constructing NAS_Bench: {}'.format(cfg.NAS))

        spec = ModelSpec(cfg.NAS.MATRIX, cfg.NAS.OPS)
        # Stem
        out_channels = cfg.NAS.WIDTH
        if 'cifar' in cfg.TRAIN.DATASET:
            logger.info('Using CIFAR10 stem')
            self.stem = base_ops.Conv3x3BnRelu(cfg.MODEL.INPUT_CHANNELS, out_channels)
        else:
            logger.info('Using ImageNet stem')
            self.stem = nn.Sequential(
                base_ops.conv_bn_relu(cfg.MODEL.INPUT_CHANNELS, out_channels // 2, 3, 2, 1),
                base_ops.conv_bn_relu(out_channels // 2, out_channels, 3, 2, 1),
                base_ops.conv_bn_relu(out_channels, out_channels, 3, 2, 1),
            )
        # Cells
        in_channels = out_channels
        self.layers = nn.ModuleList([])
        for stack_num in range(cfg.NAS.NUM_STACKS):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)
                out_channels *= 2
            for module_num in range(cfg.NAS.NUM_MODULES_PER_STACK):
                cell = Cell(spec, in_channels, out_channels)
                self.layers.append(cell)
                in_channels = out_channels
        # Classifier
        num_classes = cfg.MODEL.NUM_CLASSES
        self.classifier = Classifier(out_channels, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        x = Preprocess(x)
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x, input_size)
        return x


class Cell(nn.Module):
    def __init__(self, spec, in_channels, out_channels):
        super(Cell, self).__init__()
        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]
        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = compute_vertex_channels(
            in_channels, out_channels, self.spec.matrix)
        
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices - 1):
            op = base_ops.OP_MAP[self.spec.ops[t]](self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(projection(in_channels, self.vertex_channels[t]))
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]
        final_concat_in = []
        for t in range(1, self.num_vertices - 1):
            # Create interior connections, truncating if necessary
            add_in = [truncate(tensors[src], self.vertex_channels[t])
                      for src in range(1, t) if self.spec.matrix[src, t]]
            # Create add connection from projected input
            if self.spec.matrix[0, t]:
                add_in.append(self.input_op[t](tensors[0]))
            vertex_input = sum(add_in)
            # Perform op at vertex t
            vertex_value = self.vertex_op[t](vertex_input)
            tensors.append(vertex_value)
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(tensors[t])
        # Construct final output tensor by concating all fan-in and adding input.
        if not final_concat_in:
            # No interior vertices, input directly connected to output
            assert self.spec.matrix[0, self.num_vertices - 1]
            outputs = self.input_op[self.num_vertices - 1](tensors[0])
        else:
            if len(final_concat_in) == 1:
                outputs = final_concat_in[0]
            else:
                outputs = torch.cat(final_concat_in, 1)
            if self.spec.matrix[0, self.num_vertices - 1]:
                outputs += self.input_op[self.num_vertices - 1](tensors[0])
        return outputs


def projection(in_channels, out_channels):
    return base_ops.Conv1x1BnRelu(in_channels, out_channels)


def truncate(inputs, channels):
    input_channels = inputs.size(1)
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        input_channels: input channel count.
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels
