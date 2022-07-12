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
                 num_frames=32, batch_size=32, skip_rate=2):

        self.root = root
        self.train = train
        self.transform = transform
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.skip_rate = skip_rate
        self.video_list = self.get_list()
        self.cent_dict = self.get_list_1() 
        self.aerial_data = self.get_list_2()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        #video_name, min_lon, min_lat, max_lon, max_lat = self.video_list[idx].split(' ') # line b1d7b3ac-5744370e 34.5876553841 31.5165259363 34.5905680117 31.5185169736 #name minlon minlat mlon mlat
        #video_name, frames = self.video_list[idx].split(' ') # line b1d7b3ac-5744370e 1200 #name frames, # name minlon minlat mlon mlat
        #video_name = self.video_list[idx].split(' ')[0] # line b1d7b3ac-5744370e 34.5876553841 31.5165259363 34.5905680117 31.5185169736 #name minlon minlat mlon mlat
        video_name = self.video_list[idx] # line b1d7b3ac-5744370e
        #video_name, start_sec, sat_tile = self.video_list[idx].split(' ')
        #print(self.root)
        #print(video_name, frames)
        

        #video = self.get_sample(video_name, int(start_sec))
        #video = self.get_sample(video_name) # for ideal/gt list # can comment get_list_1
        video = self.get_sample_1(video_name) # for predicted list

        return video

    def get_sample(self, video_name):
        
        fps = 30
        skip_rate = self.skip_rate
        mode = 'train'
        if not self.train:
            mode = 'val'

        #aerial_sq = random.sample(range(0, 40), self.num_frames)
        #skip_rate = 4
        #aerial_sq = [1,2,3,4,5,6,7,8] * skip_rate
        #aerial_sq = random.sample(range(0, 40), 8)
        #print('aerial_sq',aerial_sq)
        root='/home/c3-0/shruti/data/bdd_vgl/satelite_data'
        #anchor = random.randint(0,4)
        anchor = 0
        
        sq_container = []
        flag=0
        path=os.path.join(root, 'val_gps', video_name, '00.jpeg')
        
        if not os.path.exists(path):
            
            print('00.jpeg does not exist:', path)
            
        #for item in aerial_sq:
        for item in range(anchor, anchor+self.num_frames):
            #item=-1
            image_path = os.path.join(root, 'val_gps', video_name, '%02d.jpeg' %item)
            if os.path.exists(image_path):
                current_image = Image.open(image_path).convert('RGB')
                sq_container.append(current_image)
                path = image_path
                
            else:
                flag+=1
                # print(flag)
        if flag>0:
            # print(flag)
            for i in range(flag):
                #print('path used', path)
                #print('image_path',image_path)
                current_image1 = Image.open(path).convert('RGB')
                sq_container.append(current_image1)
                # assumed that atleast one of the aerial image was found for a video

        if self.transform is not None:
            # self.transform.randomize_parameters()
            clip = [self.transform(img) for img in sq_container]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip


    def get_sample_1(self, video_name):
        
        
        #img_names = [line.rstrip('\n') for line in open(pred_list, 'r')]
        aerial_list = []
        #print(video_name)
        #video_name1 = video_name[0]
        video_name1 = video_name
        
        aerial_list = self.aerial_data[video_name1]
        #print(len(aerial_list))
        
        if len(aerial_list)>= self.num_frames:
            #aerial_sq = random.sample(aerial_list, self.num_frames)
            aerial_sq = aerial_list[0:self.num_frames]
        
        else:
            aerial_sq = random.choices(aerial_list, k=self.num_frames)
            
        root='/home/c3-0/shruti/data/bdd_vgl/satelite_data'
        
        sq_container = []
        flag = 0
        
        path=os.path.join(root, 'val_gps', video_name1, '00.jpeg')
        
        if not os.path.exists(path):
            
            print('00.jpeg does not exist:', path)
            
        for item in aerial_sq:
            #item=-1
            video_name1 = item.split('_')[0]
            
            item1 = self.cent_dict[item]
            #image_path = os.path.join(root, 'val_gps', video_name, '%02d.jpeg' %item1)
            image_path = os.path.join(root, 'val_gps', item1)
            if os.path.exists(image_path):
                current_image = Image.open(image_path).convert('RGB')
                sq_container.append(current_image)
                path = image_path
                
            else:
                flag+=1
                # print(flag)
        if flag>0:
            # print(flag)
            for i in range(flag):
                #print('path used', path)
                #print('image_path',image_path)
                current_image1 = Image.open(path).convert('RGB')
                sq_container.append(current_image1)
                # assumed that atleast one of the aerial image was found for a video

        if self.transform is not None:
            # self.transform.randomize_parameters()
            clip = [self.transform(img) for img in sq_container]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip
        
    def get_list(self):

        #test_list = os.path.join(self.root, 'val_gps_day.list')
        #test_list = os.path.join(self.root, 'val.list') # val video list # line b1d7b3ac-5744370e frames
        test_list = os.path.join(self.root, 'val_day_vid_1.list') # val video list # line b1d7b3ac-5744370e\n
        # test_list = os.path.join(self.root, 'val_day_vid_tiny.list')
    
        img_names = [line.rstrip('\n') for line in open(test_list, 'r')]

        return img_names
        
    def get_list_1(self):

        cent_list = '/home/c3-0/shruti/data/maps/code_day_unsup_center_aug_Laerial/retrieved/dict_centered.json'
        cent_dict = json.load(open(cent_list, 'r'))
        

        return cent_dict
    
    def get_list_2(self):

        pred_list = '/home/c3-0/shruti/data/maps/code_day_unsup_center_aug_Laerial/retrieved/gt_retrieved.json'
        aerial_data = json.load(open(pred_list, 'r'))      

        return aerial_data


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
        #img_name = self.gallery_list[idx].split()[0] ## gallery list line b1d7b3ac-5744370e 34.5876553841 31.5165259363 34.5905680117 31.5185169736
        img_name = self.gallery_list[idx] ## gallery list line b1d7b3ac-5744370e
        #img_name1 = img_name.split('/')[0]
        mode = 'val_hr'

        #sat_image = self.get_sample(img_name)
        sat_image = self.get_sat_imageA(mode, img_name)

        return sat_image


    def get_sat_imageA(self, mode, video_name):
        sat_image = None
        # print(start_gps)
        # print(mode)
        #image_name = str(start_gps) + '.jpeg'
        # print('3:', mode, video_name, start_gps)
        #print('video_name', video_name)
        root='/home/c3-0/shruti/data/bdd_vgl/satelite_data'
        s_path = os.path.join(root, mode, video_name +'.jpeg')
        img_path = s_path
        #img_path = os.path.join(s_path, '%02d.jpeg' % start_gps) 
        #vidcap = cv2.VideoCapture(img_path)
        #success,image = vidcap.read()
        sat_image = Image.open(img_path).convert('RGB') #1792x1792
        #sat_image = cv2.resize(image, (256, 256))
        #cv2.imshow('sample', sat_image)
        
        if self.transform is not None:
            # self.transform.randomize_parameters()
            sat_image = self.transform(sat_image)

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
        #gallery_list = os.path.join(self.root, 'gallery_gps_day_center.list') # centered day list
        #gallery_list = os.path.join(self.root, 'val.list') # val video list # line b1d7b3ac-5744370e 1200
        gallery_list = os.path.join(self.root, 'val_day_vid_1.list') # val video list # line b1d7b3ac-5744370e
        if self.full_gallery:
            gallery_list = os.path.join(self.root, 'gallery_F.list')
            
    
        img_names = [line.rstrip('\n') for line in open(gallery_list, 'r')]

        return img_names


