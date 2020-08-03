#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
from skimage import color

import pycls.datasets.transforms as transforms
from pycls.core.config import cfg


def prepare_rot(im, dataset, split, mean, sd, eig_vals=None, eig_vecs=None):
    im = prepare_im(im, dataset, split, mean, sd, eig_vals, eig_vecs)
    rot_im = []
    for i in range(4):
        rot_im.append(np.rot90(im, i, (1, 2)))
    im = np.stack(rot_im, axis=0)
    label = np.array([0, 1, 2, 3])
    return im, label


def prepare_col(im, dataset, split, nbrs, mean, sd, eig_vals=None, eig_vecs=None):
    if "cifar" not in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            if "imagenet" in dataset:
                im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            elif "cityscapes" in dataset:
                random_scale = np.power(2, -1 + 2 * np.random.uniform())
                random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
                im = transforms.scale(random_size, im)
                im = transforms.random_crop(im, train_size, order="HWC")
        else:
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    if split == "train":
        im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")  # Best before rgb2lab because otherwise need to flip together with label
    im_lab = color.rgb2lab(im.astype(np.uint8, copy=False))
    im = transforms.HWC2CHW(im_lab[:, :, 0:1]).astype(np.float32, copy=False)
    # Ad hoc normalization of the L channel
    im = im / 100.0
    im = im - np.mean(mean)
    im = im / np.mean(sd)
    label = nbrs.kneighbors(im_lab[:, :, 1:].reshape(-1, 2),
            return_distance=False).reshape(im_lab.shape[0], im_lab.shape[1])
    return im, label


def prepare_jig(im, dataset, split, perms, mean, sd, eig_vals=None, eig_vecs=None):
    if "cifar" not in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            if "imagenet" in dataset:
                target_size = cfg.TEST.IM_SIZE
            elif "cityscapes" in dataset:
                random_scale = np.power(2, -1 + 2 * np.random.uniform())
                target_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
            im = transforms.scale(target_size, im)
            im = transforms.random_crop(im, train_size, order="HWC")
        else:
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    if split == "train":
        im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        if np.random.uniform() < cfg.TRAIN.GRAY_PERCENTAGE:
            im = color.rgb2gray(im.astype(np.uint8, copy=False)) * 255.0
            im = np.expand_dims(im, axis=2)
            im = np.tile(im, (1, 1, 3))
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    if "cifar" not in dataset:
        im = im[:, :, ::-1]  # RGB -> BGR
        # PCA jitter
        if split == "train":
            im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)
    # Random permute
    label = np.random.randint(len(perms))
    perm = perms[label]
    # Crop tiles
    psz = int(cfg.TRAIN.IM_SIZE / cfg.JIGSAW_GRID)  # Patch size
    tsz = int(psz * 0.76)  # Tile size; int(85 * 0.76) = 64
    tiles = np.zeros((cfg.JIGSAW_GRID ** 2, 3, tsz, tsz)).astype(np.float32)
    for i in range(cfg.JIGSAW_GRID):
        for j in range(cfg.JIGSAW_GRID):
            patch = im[:, psz * i:psz * (i+1), psz * j:psz * (j+1)]
            # Gap
            h = np.random.randint(psz - tsz + 1)
            w = np.random.randint(psz - tsz + 1)
            tile = patch[:, h:h+tsz, w:w+tsz]
            # Normalization
            mu, sigma = np.mean(tile), np.std(tile)
            tile = tile - mu
            tile = tile / sigma
            tiles[perm[cfg.JIGSAW_GRID * i + j]] = tile
    return tiles, label


def prepare_im(im, dataset, split, mean, sd, eig_vals=None, eig_vecs=None):
    if "imagenet" in dataset:
        # Train and test setups differ
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            # Scale and aspect ratio then horizontal flip
            im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale and center crop
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    elif "cityscapes" in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            # Scale
            random_scale = np.power(2, -1 + 2 * np.random.uniform())
            random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
            im = transforms.scale(random_size, im)
            # Crop
            im = transforms.random_crop(im, train_size, order="HWC")
            # Flip
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            # Crop
            im = transforms.center_crop(train_size, im)
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    if "cifar" not in dataset:
        im = im[:, :, ::-1]  # RGB -> BGR
        # PCA jitter
        if split == "train":
            im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)
    if "cifar" in dataset:
        if split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(im=im, size=cfg.TRAIN.IM_SIZE, pad_size=4)  # Best after color_norm because of zero padding
    return im


def prepare_seg(im, label, split, mean, sd, eig_vals=None, eig_vecs=None):
    if split == "train":
        train_size = cfg.TRAIN.IM_SIZE
        # Scale
        random_scale = np.power(2, -1 + 2 * np.random.uniform())
        random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
        im = transforms.scale(random_size, im)
        label = transforms.scale(random_size, label, interpolation=cv2.INTER_NEAREST, dtype=np.int64)
        # Crop
        h, w = im.shape[:2]
        y = 0
        if h > train_size:
            y = np.random.randint(0, h - train_size)
        x = 0
        if w > train_size:
            x = np.random.randint(0, w - train_size)
        im = im[y : (y + train_size), x : (x + train_size), :]
        label = label[y : (y + train_size), x : (x + train_size)]
        # Flip
        if np.random.uniform() < 0.5:
            im = im[:, ::-1, :].copy()
            label = label[:, ::-1].copy()
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    im = im[:, :, ::-1]  # RGB -> BGR
    # PCA jitter
    if split == "train":
        im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)
    if split != "train":
        # 1025 x 2049; cfg.TEST.IM_SIZE is not used here
        im = np.pad(im, ((0, 0), (0, 1), (0, 1)), "constant", constant_values=0)  # Best after color_norm because of zero padding
        label = np.pad(label, ((0, 1), (0, 1)), "constant", constant_values=255)
    return im, label
