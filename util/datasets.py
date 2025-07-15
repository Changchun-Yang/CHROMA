# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .pretrain_chr_dataloader import ChrDataset
from torch.utils.data import WeightedRandomSampler
import torch
import random


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# def build_chr_dataset(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     root = os.path.join(os.path.join(args.data_path, 'private'), is_train)
#     if is_train=='train':
#         # fix me: consider to introduce the re_prob, re_mode=args.remode, re_count=args.recount,
#         dataset = ChrDataset(
#             root, 
#             data_size=(args.input_size, args.input_size), 
#             patch_size=int(args.model.split('patch')[-1]), 
#             transform=True,
#             w_flip=True, w_rotate=True, w_scale=False,  
#             normalize={
#             'mean': mean,  # ImageNet or custom means
#             'std': std    # ImageNet or custom stds
#             }
#         )
#     else:
#         dataset = ChrDataset(
#             root, 
#             data_size=(args.input_size, args.input_size), 
#             patch_size=int(args.model.split('patch')[-1]), 
#             transform=True,
#             w_flip=False, w_rotate=False, w_scale=False,  
#             normalize={
#             'mean': mean,  # ImageNet or custom means
#             'std': std    # ImageNet or custom stds
#             }
#         )
#     return dataset

# for different chr data size, we need padding
def build_chr_dataset(is_train, args):
    transform = build_chr_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_chr_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    # if is_train=='train':
    #     # this should always dispatch to transforms_imagenet_train
    #     transform = create_transform(
    #         input_size=args.input_size,
    #         is_training=True,
    #         # color_jitter=args.color_jitter,
    #         # auto_augment=args.aa,
    #         interpolation='bicubic',
    #         # re_prob=args.reprob,
    #         # re_mode=args.remode,
    #         # re_count=args.recount,
    #         mean=mean,
    #         std=std,
    #     )
    #     if args.w_pad:
    #         transform.transforms.insert(0, PadToSize(args.input_size, w_randp=False))
    #     return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(args.input_size / crop_pct)
    if args.w_pad:
        t.append(PadToSize(args.input_size, w_randp=False))
    t.append(
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    # t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def pad_image(image, target_size, w_randp=False):
    size = image.size
    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    elif isinstance(target_size, (tuple, list)):
        target_size = tuple(target_size)
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    if w_randp:
        max_x = target_size[0] - size[0]
        max_y = target_size[1] - size[1]
        upper_left_x = random.randint(0, max_x) if max_x > 0 else 0
        upper_left_y = random.randint(0, max_y) if max_y > 0 else 0
        new_image.paste(image, (upper_left_x, upper_left_y))
        new_bbox = [upper_left_x, upper_left_y, size[0], size[1]]
    else:
        upper_left_x = max( (target_size[0] - size[0]) // 2 , 0)
        upper_left_y = max( (target_size[1] - size[1]) // 2 , 0)
        new_image.paste(image, (upper_left_x, upper_left_y))

        new_bbox = [upper_left_x, upper_left_y, size[0], size[1]]

    return new_image


class PadToSize(object):
    def __init__(self, size, w_randp=False):
        self.size = size
        self.w_randp = w_randp
    
    def __call__(self, image):
        return pad_image(image, self.size, self.w_randp)

