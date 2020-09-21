#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cityscapes dataset."""

import os
import re
import glob
from sklearn.neighbors import NearestNeighbors

import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg
from pycls.datasets.prepare import prepare_rot
from pycls.datasets.prepare import prepare_col
from pycls.datasets.prepare import prepare_jig
from pycls.datasets.prepare import prepare_seg


logger = logging.get_logger(__name__)
folder = os.path.dirname(os.path.realpath(__file__))

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


class Cityscapes(torch.utils.data.Dataset):
    """Cityscapes dataset."""

    def __init__(self, data_path, split, portion, side):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for Cityscapes".format(split)
        logger.info("Constructing Cityscapes {}...".format(split))
        self._data_path, self._split = data_path, split
        self._portion, self._side = portion, side
        if cfg.TASK == 'col':
            # Color centers in ab channels; numpy array; shape (313, 2)
            self._pts = np.load(os.path.join(folder, "files", "pts_in_hull.npy"))
            self._nbrs = NearestNeighbors(n_neighbors=1).fit(self._pts)
        elif cfg.TASK == 'jig':
            assert cfg.JIGSAW_GRID in [2, 3]
            if cfg.JIGSAW_GRID == 3:
                assert cfg.MODEL.NUM_CLASSES in [1000, 2000]
                if cfg.MODEL.NUM_CLASSES == 1000:
                    # Jigsaw permutations; numpy array; shape (1000, 9)
                    fname = "hamming_perms_1000_patches_9_max.npy"
                else:
                    # Jigsaw permutations; numpy array; shape (2000, 9)
                    fname = "hamming_perms_2000_patches_9_max_avg.npy"
            else:
                assert cfg.MODEL.NUM_CLASSES == 24
                # Jigsaw permutations; numpy array; shape(24, 4)
                fname = "permutations_24.npy"
            self._perms = np.load(os.path.join(folder, "files", fname))
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        image_dir = os.path.join(self._data_path, "leftImg8bit", self._split)
        assert os.path.exists(image_dir), "{} dir not found".format(image_dir)
        label_dir = os.path.join(self._data_path, "gtFine", self._split)
        assert os.path.exists(label_dir), "{} dir not found".format(label_dir)
        self.images = sorted(glob.glob(os.path.join(image_dir, "*", "*_leftImg8bit.png")))
        self.labels = []
        for image_fname in self.images:
            image_fname = image_fname.split("/")[-1].split("leftImg8bit")[0]
            label_fname = os.path.join(label_dir, image_fname.split("_")[0], image_fname + "gtFine_labelTrainIds.png")
            assert os.path.exists(label_fname), "{} not found".format(label_fname)
            self.labels.append(label_fname)
        # Construct the image db
        self._imdb = []
        for im_path, label_path in zip(self.images, self.labels):
            self._imdb.append({"im_path": im_path, "class": label_path})
        if self._portion:
            # Shuffle so that partition is not correlated with class
            np.random.seed(cfg.RNG_SEED)
            np.random.shuffle(self._imdb)
            pos = int(self._portion * len(self._imdb))
            if self._side == "l":
                self._imdb = self._imdb[:pos]
            else:  # self._side == "r"
                self._imdb = self._imdb[pos:]
        logger.info("Number of images: {}".format(len(self._imdb)))

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        im = im.astype(np.float32, copy=False)
        im = im[:, :, ::-1]  # HWC, BGR -> HWC, RGB
        if cfg.TASK == 'rot':
            im, label = prepare_rot(im,
                                    dataset="cityscapes",
                                    split=self._split,
                                    mean=_MEAN,
                                    sd=_SD,
                                    eig_vals=_EIG_VALS,
                                    eig_vecs=_EIG_VECS)
        elif cfg.TASK == 'col':
            im, label = prepare_col(im,
                                    dataset="cityscapes",
                                    split=self._split,
                                    nbrs=self._nbrs,
                                    mean=_MEAN,
                                    sd=_SD,
                                    eig_vals=_EIG_VALS,
                                    eig_vecs=_EIG_VECS)
        elif cfg.TASK == 'jig':
            im, label = prepare_jig(im,
                                    dataset="cityscapes",
                                    split=self._split,
                                    perms=self._perms,
                                    mean=_MEAN,
                                    sd=_SD,
                                    eig_vals=_EIG_VALS,
                                    eig_vecs=_EIG_VECS)
        else:
            # Prepare the image for training / testing
            label = cv2.imread(self._imdb[index]["class"], cv2.IMREAD_UNCHANGED)
            label = label.astype(np.int64, copy=False)
            im, label = prepare_seg(im,
                                    label,
                                    split=self._split,
                                    mean=_MEAN,
                                    sd=_SD,
                                    eig_vals=_EIG_VALS,
                                    eig_vecs=_EIG_VECS)
        return im.copy(), label

    def __len__(self):
        return len(self._imdb)
