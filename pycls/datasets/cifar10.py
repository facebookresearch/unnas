#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg
from pycls.datasets.prepare import prepare_rot
from pycls.datasets.prepare import prepare_col
from pycls.datasets.prepare import prepare_jig
from pycls.datasets.prepare import prepare_im


logger = logging.get_logger(__name__)
folder = os.path.dirname(os.path.realpath(__file__))

# Per-channel mean and SD values in BGR order
_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
_SD = [x / 255.0 for x in [63.0, 62.1, 66.7]]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split, portion=None, side=None):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._data_path, self._split = data_path, split
        self._portion, self._side = portion, side
        if cfg.TASK == 'col':
            # Color centers in ab channels; numpy array; shape (313, 2)
            self._pts = np.load(os.path.join(folder, "files", "pts_in_hull.npy"))
            self._nbrs = NearestNeighbors(n_neighbors=1).fit(self._pts)
        elif cfg.TASK == 'jig':
            assert cfg.JIGSAW_GRID == 2
            assert cfg.MODEL.NUM_CLASSES == 24
            # Jigsaw permutations; numpy array; shape (24, 4)
            self._perms = np.load(os.path.join(folder, "files", "permutations_24.npy"))
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            with open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            inputs.append(data[b"data"])
            labels += data[b"labels"]
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        if self._portion:
            # CIFAR-10 data are random, so no need to shuffle
            pos = int(self._portion * len(inputs))
            if self._side == "l":
                return inputs[:pos], labels[:pos]
            else:  # self._side == "r"
                return inputs[pos:], labels[pos:]
        else:
            return inputs, labels

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = transforms.CHW2HWC(im)  # CHW, RGB -> HWC, RGB
        if cfg.TASK == 'rot':
            im, label = prepare_rot(im,
                                    dataset="cifar10",
                                    split=self._split,
                                    mean=_MEAN,
                                    sd=_SD)
        elif cfg.TASK == 'col':
            im, label = prepare_col(im,
                                    dataset="cifar10",
                                    split=self._split,
                                    nbrs=self._nbrs,
                                    mean=_MEAN,
                                    sd=_SD)
        elif cfg.TASK == 'jig':
            im, label = prepare_jig(im,
                                    dataset="cifar10",
                                    split=self._split,
                                    perms=self._perms,
                                    mean=_MEAN,
                                    sd=_SD)
        else:
            im = prepare_im(im,
                            dataset="cifar10",
                            split=self._split,
                            mean=_MEAN,
                            sd=_SD)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
