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


class bdd_vgl(Dataset):
    def __init__(self, root = '', train=True, v_transform=None,
                 i_transform=None,
                 num_frames=32, batch_size=32, skip_rate=2):

        self.root = root
        self.train = train
        self.video_list = self.get_list()
        self.v_transform = v_transform
        self.i_transform = i_transform
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.skip_rate = skip_rate

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name, num_frames = self.video_list[idx].split(' ')

        #video, sat_image = self.get_sample(video_name, int(num_frames))
        sq_aerial, Aerial_image = self.get_sample(video_name, int(num_frames))
        # print(sq_aerial.shape)
        # print(Aerial_image.shape)

        label = idx%self.batch_size #hn
        
        # get the hard negative
        #video_hn, sat_image_hn = self.get_sample_hn(video_name, int(num_frames)) #hn

        # label_hn = idx%self.batch_size + self.batch_size

        #videos = torch.stack([video, video_hn], dim=0) #hn
        #images = torch.stack([sat_image, sat_image_hn], dim=0) #hn
        # labels = torch.stack([label, label_hn], dim=0)

        #return videos, images# , label, label_hn #hn

        return sq_aerial, Aerial_image, label

    def get_start_sec(self, meta_data, gps_seq, clip_len):

        # select random gps

        start_gps = random.randrange(1, len(gps_seq)-1)

        t0 = meta_data['startTime']
        t1 = gps_seq[start_gps]['timestamp']

        start_sec = int(round((t1 - t0)/1000))
        if start_sec > clip_len-2:
            start_sec = clip_len - 2

        if t0 > t1:
            start_gps, start_sec = self.get_start_sec(meta_data, gps_seq[1:], clip_len)
            start_gps += 1

        return start_gps, start_sec


    def get_start_sec_hn(self, meta_data, gps_seq, clip_len):

        # select random gps
        start_gps = 0

        mid_len = int(len(gps_seq)/2)

        if self.start_gps > mid_len:
            try:
                start_gps = random.randrange(1, mid_len-5)
            except:
                start_gps = 1
        else:
            try:
                start_gps = random.randrange(mid_len+3, len(gps_seq)-1)
            except:
                start_gps = len(gps_seq)-2

        t0 = meta_data['startTime']
        t1 = gps_seq[start_gps]['timestamp']

        start_sec = int(round((t1 - t0)/1000))
        if start_sec > clip_len-2:
            start_sec = clip_len - 2

        if t0 > t1:
            start_gps, start_sec = self.get_start_sec_hn(meta_data, gps_seq[1:], clip_len)
            start_gps += 1

        return start_gps, start_sec
        
    def get_sample_hn(self, video_name, num_frames):
        
        fps = 30
        skip_rate = self.skip_rate
        mode = 'train'
        if not self.train:
            mode = 'val'

        clip_len = int(num_frames/fps)

        # load meta data
        meta_data_path = os.path.join(self.root, 'info', mode, video_name+'.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        gps_seq = meta_data['locations']

        start_gps, start_sec = self.get_start_sec_hn(meta_data, gps_seq, clip_len)

        total_frames = self.num_frames*skip_rate

        if total_frames > fps:
            skip_rate = skip_rate -1
            if skip_rate == 0:
                skip_rate = 1
            total_frames = 16*skip_rate

        try:
            start_frame = random.randint(0, fps - total_frames) ## 32, 16 frames
        except:
            start_frame = 0

        start_frame = start_sec*fps + start_frame - 15 # centralize the gps location
        #start_frame = start_sec*fps + start_frame # centralize the gps location

        if start_frame < 0:
            start_frame = 0

        video_container = []
        for item in range(start_frame, start_frame + total_frames, skip_rate):
            image_name = str(item).zfill(4) + '.png'
            image_path = os.path.join(self.root, 'video_data', mode, video_name+'.mov', image_name)
            current_image = Image.open(image_path).convert('RGB')
            video_container.append(current_image)

        # get the corresponding satelite image
        # print('1:', mode, video_name, start_gps)
        # satelite_image = self.get_sat_image(video_name, gps_seq, start_gps)
        satelite_image = self.get_sat_image(mode, video_name, start_gps)

        if self.i_transform is not None:
            # self.i_transform.randomize_parameters()
            clip = [self.v_transform(img) for img in video_container]

        if self.i_transform is not None:
            # self.i_transform.randomize_parameters()
            satelite_image = self.i_transform(satelite_image)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if start_sec > clip_len:
            print(start_sec)
            print(video_name)
            print(t0)
            print(t1)
            print(num_frames)

        # print(video_name)
        # img_name = os.path.join('./test', video_name+'.jpeg')
        # satelite_image.save()

        return clip, satelite_image
        
    def get_sample(self, video_name, num_frames):
        
        fps = 30
        #skip_rate = self.skip_rate
        mode = 'train'
        if not self.train:
            mode = 'val'

        #clip_len = int(num_frames/fps)
        
        
        # load meta data
        #meta_data_path = os.path.join(self.root, 'info', mode, video_name+'.json')
        #meta_data = json.load(open(meta_data_path, 'r'))
        #gps_seq = meta_data['locations']

        #start_gps, start_sec = self.get_start_sec(meta_data, gps_seq, clip_len)
        #self.start_gps = start_gps

        #total_frames = self.num_frames*skip_rate

        #if total_frames > fps:
        #    skip_rate = skip_rate -1
        #    if skip_rate == 0:
        #        skip_rate = 1
        #    total_frames = 16*skip_rate

        #try:
        #    start_frame = random.randint(0, fps - total_frames) ## 32, 16 frames
        #except:
        #    start_frame = 0

        #start_frame = start_sec*fps + start_frame - 15 # centralize the gps location
        #start_frame = start_sec*fps + start_frame

        #if start_frame < 0:
        #    start_frame = 0

        #video_container = []
        #for item in range(start_frame, start_frame + total_frames, skip_rate):
        #    image_name = str(item).zfill(4) + '.png'
        #    image_path = os.path.join(self.root, 'video_data', mode, video_name+'.mov', image_name)
        #    current_image = Image.open(image_path).convert('RGB')
        #    video_container.append(current_image)
        
        
        #aerial_sq = random.sample(range(0, 40), self.num_frames)
        #aerial_sq = random.sample(range(0, 40), 8)
        #print('aerial_sq',aerial_sq)
        root='/home/c3-0/shruti/data/bdd_vgl/satelite_data'
        anchor = random.randint(0,4)
        
        sq_container = []
        flag=0
        path=os.path.join(root, 'train_gps', video_name, '00.jpeg')
        #for item in aerial_sq:
        for item in range(anchor, anchor+self.num_frames):
            #item=-1
            image_path = os.path.join(root, 'train_gps', video_name, '%02d.jpeg' %item)
            #print(item)
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
            
        # get the corresponding satelite image
        # print('2:', mode, video_name, start_gps)
        #satelite_image = self.get_sat_image(mode, video_name, start_gps)
        #root='/home/c3-0/shruti/data/bdd_vgl/satelite_data/val_hr'
        mode='train_hr'
        satelite_image = self.get_sat_imageA(mode, video_name) #256x256

        if self.i_transform is not None:
            # self.i_transform.randomize_parameters()
            clip = [self.i_transform(img) for img in sq_container]

        if self.i_transform is not None:
            # self.i_transform.randomize_parameters()
            satelite_image = self.i_transform(satelite_image)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, satelite_image


    def get_sat_image_v1(self, video_name, gps_seq, start_sec):
        
        mode = 'train'
        if not self.train:
            mode = 'val'

        gps = gps_seq[start_sec]

        lat = gps['latitude']
        lon = gps['longitude']

        x_tile, y_tile = self.get_tile_loc(lat, lon)

        image_name = str(x_tile) + '_' + str(y_tile) + '.jpeg'
        image_path = os.path.join(self.root, 'satelite_data', mode, video_name, image_name)

        sat_image = Image.open(image_path).convert('RGB')

        return sat_image
        
    def get_sat_imageA(self, mode, video_name):
        sat_image = None
        # print(start_gps)
        # print(mode)
        #image_name = str(start_gps) + '.jpeg'
        # print('3:', mode, video_name, start_gps)
        root='/home/c3-0/shruti/data/bdd_vgl/satelite_data'
        #s_path = os.path.join(root, mode, video_name + '.jpeg')
        s_path = os.path.join(self.root, 'satelite_data', mode, video_name + '.jpeg')
        img_path = s_path
        #img_path = os.path.join(s_path, '%02d.jpeg' % start_gps) 
        #vidcap = cv2.VideoCapture(img_path)
        #success,image = vidcap.read()
        sat_image = Image.open(img_path).convert('RGB') #1792x1792
        #sat_image = cv2.resize(image, (256, 256))
        #cv2.imshow('sample', sat_image)

        return sat_image
        
    def get_sat_image(self, mode, video_name, start_gps):
        sat_image = None
        # print(start_gps)
        # print(mode)
        image_name = str(start_gps) + '.jpeg'
        # print('3:', mode, video_name, start_gps)
        s_path = os.path.join(self.root, 'satelite_data', mode+'_gps', video_name)
        img_path = os.path.join(s_path, '%02d.jpeg' % start_gps) 
        sat_image = Image.open(img_path).convert('RGB')

        return sat_image

    def ret_lat_lon(self,x_tyle,y_tyle):                                                                                                                          
        # This function returns the lat, lon as a tuple
        # Takes x_tyle, y_tyle and zoom_level
        n = 2**19
        lon_deg = int(x_tyle)/n * 360.0 - 180.0
        lat_rad = math.atan(math.asinh(math.pi * (1 - 2 * int(y_tyle)/n)))
        lat_deg = math.degrees(lat_rad)    
        # lat_deg = lat_rad * 180.0 / math.pi

        return lat_deg, lon_deg


    def get_tile_loc(self, lat_deg, lon_deg):
        # changes for 0.0005
        # This function returns the tilex and tiley in tuple
        # Takes latitude, longitude and zoom_level
        n = 2**19
        xtile = n * ((lon_deg + 180) / 360)
        # lat_rad = math.radians(lat_deg)
        lat_rad = lat_deg * math.pi / 180.0
        ytile = n * (1 - (math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi)) / 2

        return int(xtile), int(ytile)


    def get_list(self):
        mode = 'train'
        if not self.train:
            mode = 'val'

        train_list = os.path.join(self.root, mode+'_day.list')
    
        video_names = [line.rstrip('\n') for line in open(train_list, 'r')]

        return video_names


