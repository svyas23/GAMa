import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from .spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, RandomCrop)

from .dataset_train import bdd_vgl
from .dataset_test_C import bdd_vgl_gallery, bdd_vgl_test
import torch

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def set_loader_test(opt):
    # construct data loader
    
    test_transform_video = Compose([
        CenterCrop(size=(opt.video_size_h, opt.video_size_w)),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    test_dataset = bdd_vgl_test(root=opt.data_folder,
                            transform=test_transform_video, 
                            batch_size=opt.batch_size)

    test_loader = DataLoader(
        test_dataset,
        sampler=None,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        pin_memory=True)

    return test_loader

def set_loader_gallery(opt):
    # construct data loader
    
    gallery_transform_image = Compose([
        Scale(size=(256,256)),
        CenterCrop(size=(opt.image_size_h, opt.image_size_w)),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    gallery_dataset = bdd_vgl_gallery(root=opt.data_folder,
                            transform=gallery_transform_image, 
                            batch_size=opt.batch_size,
                            full_gallery=opt.full_gallery)

    gallery_loader = DataLoader(
        gallery_dataset,
        sampler=None,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        pin_memory=True)

    return gallery_loader

def set_loader(opt):
    # construct data loader
    
    train_transform_video = Compose([
        RandomCrop(size=(opt.video_size_h, opt.video_size_w)),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    train_transform_image = Compose([
        RandomCrop(size=(opt.image_size_h, opt.image_size_w)),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    train_dataset = bdd_vgl(root=opt.data_folder,
                            v_transform=train_transform_video,
                            i_transform=train_transform_image, 
                            batch_size=opt.batch_size)

    train_sampler = RandomSampler 

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=True,
        pin_memory=True)

    return train_loader

