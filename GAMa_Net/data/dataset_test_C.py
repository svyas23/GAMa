from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import random
import pickle, h5py
import cv2
import json
import torch
import torch
from torch.autograd import Variable
import json
from skimage.transform import resize
from skimage import img_as_bool
from PIL import Image
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


class bdd_vgl_test(Dataset):
    def __init__(self, root = '', train=False, transform=None,
                 num_frames=8, batch_size=32, skip_rate=2):

        self.root = root
        self.train = train
        self.transform = transform
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.skip_rate = skip_rate
        self.video_list = self.get_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name, start_sec, sat_tile = self.video_list[idx].split(' ')

        video = self.get_sample(video_name, int(start_sec))

        return video

    def get_sample(self, video_name, start_sec):
        
        fps = 30
        skip_rate = self.skip_rate
        mode = 'train'
        if not self.train:
            mode = 'val'

        total_frames = self.num_frames*skip_rate

        start_frame = start_sec*fps

        video_container = []
        for item in range(start_frame, start_frame + total_frames, skip_rate):
            image_name = str(item).zfill(4) + '.png'
            image_path = os.path.join(self.root, 'video_data', mode, video_name+'.mov', image_name)
            current_image = Image.open(image_path).convert('RGB')
            video_container.append(current_image)

        if self.transform is not None:
            # self.transform.randomize_parameters()
            clip = [self.transform(img) for img in video_container]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip


    def get_list(self):

        test_list = os.path.join(self.root, 'val_gps_day.list')
    
        img_names = [line.rstrip('\n') for line in open(test_list, 'r')]

        return img_names



class bdd_vgl_gallery(Dataset):
    def __init__(self, root = '', train=False, transform=None,
                 batch_size=32, full_gallery=False):

        self.root = root
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.full_gallery = full_gallery
        self.gallery_list = self.get_list()

    def __len__(self):
        return len(self.gallery_list)

    def __getitem__(self, idx):
        img_name = self.gallery_list[idx].split()[0]

        sat_image = self.get_sample(img_name)

        return sat_image


    def get_sample(self, img_name):

        mode = 'train'
        if not self.train:
            mode = 'val_gps'

        image_path = os.path.join(self.root, 'satelite_data', mode, img_name)

        sat_image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            # self.transform.randomize_parameters()
            sat_image = self.transform(sat_image)

        return sat_image


    def get_list(self):

        #gallery_list = os.path.join(self.root, 'gallery_gps_center.list') # centered list
        gallery_list = os.path.join(self.root, 'gallery_gps_day_center.list') # centered day list
        if self.full_gallery:
            gallery_list = os.path.join(self.root, 'gallery_F.list')
            
    
        img_names = [line.rstrip('\n') for line in open(gallery_list, 'r')]

        return img_names


