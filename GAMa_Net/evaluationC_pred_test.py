from __future__ import print_function

import os
import sys
import argparse
import time
import math
import json
import cv2
import pickle
from math import radians, cos, sin, asin, sqrt
import numpy as np
from scipy.spatial import distance

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_test import TestResNet
from losses import SupConLoss
from data.dataset_C import set_loader_test
import params
from shutil import copyfile

_DEBUG = False # When TRUE saves top-10 predictions

dist_dict = None
lat_lon_map = None
cnt = 0

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='bdd_vgl',
                        choices=['cifar10', 'cifar100', 'path', 'bdd_vgl'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/home/c3-0/shruti/data/bdd_vgl/', help='path to custom dataset')
    parser.add_argument('--video_size_h', type=int, default=112, help='parameter for RandomResizedCrop')
    parser.add_argument('--video_size_w', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--image_size_h', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--image_size_w', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--weights', type=str, default=params.weights, help='path to weights')
    parser.add_argument('--full_gallery', type=bool, default=False, help='whether full list or use gps')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_model(opt):
    model = TestResNet(name=opt.model)
    # print(model)

    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['model'], strict=False)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            # model.v_encoder = torch.nn.DataParallel(model.v_encoder)
            # model.i_encoder = torch.nn.DataParallel(model.i_encoder)
        model = model.cuda()
        cudnn.benchmark = True

    return model


def get_v_embeddings(opt):
        
    base_path = os.path.split(opt.weights)[0]
    embeddings_name = os.path.join(base_path, 'embeddings_videos_C.npy') 
    

    embeddings = np.load(embeddings_name)

    return embeddings

def get_embeddings(opt):

    base_path = os.path.split(opt.weights)[0]
    embeddings_name = os.path.join(base_path, 'embeddings_gps_C.npy') # centered maps with val_gps
    #embeddings_name = os.path.join(base_path, 'embeddings_gps.npy') # uncentered

    if opt.full_gallery:
        embeddings_name = os.path.join(base_path, 'embeddings_F.npy')

    embeddings = np.load(embeddings_name)

    return embeddings
    

def get_ids(opt):
    #gt_path = os.path.join(opt.data_folder, 'val_gps_center.list') # val_gps_center.list is centered on full: day+night
    gt_path = os.path.join(opt.data_folder, 'val_gps_day.list')
    
    gt_ids = [x.split()[0]+'_'+x.split()[1] for x in open(gt_path, 'r').readlines()] # B # e.g. b1c66a42-6f7d68ca_1  #val_gps_day.list
    
    gt_vids = [x.split()[0] for x in open(gt_path, 'r').readlines()] # e.g. b1c66a42-6f7d68ca
    

    #gallery_path = os.path.join(opt.data_folder, 'gallery_gps.list') # line e.g. 154379_197218 b1c66a42-6f7d68ca # uncentered
    #gallery_path = os.path.join(opt.data_folder, 'gallery_gps_center.list') # line e.g. b1c66a42-6f7d68ca/02.jpeg 1 154379_197218 #centered
    gallery_path = os.path.join(opt.data_folder, 'gallery_gps_day_center.list') # line e.g. b1c66a42-6f7d68ca/02.jpeg 1 154379_197218 # centered day list
    if opt.full_gallery:
        gallery_path = os.path.join(opt.data_folder, 'gallery_F.list')
    
    gallery_ids = [((x.split()[0]).split("/")[0]+'_'+x.split( )[1]) for x in open(gallery_path, 'r').readlines()] # B # e.g. b1c66a42-6f7d68ca_1
    
    gallery_vids = [((x.split( )[0])) for x in open(gallery_path, 'r').readlines()] # e.g. b1c66a42-6f7d68ca/02.jpeg
    
    #dictionary for correspondence:
    gallery_vids1 = {}
    for x in (open(gallery_path, 'r').readlines()):
        gallery_vids1[((x.split()[0]).split("/")[0]+'_'+x.split( )[1])] = ((x.split( )[0]))
        # dictionary form e.g. gallery_vids1[gallery_ids] = gallery_vids
    
    #[gallery_vids1[((x.split()[0]).split("/")[0]+'_'+x.split( )[1])] = ((x.split( )[0])) for x in open(gallery_path, 'r').readlines()]
    

    return gt_ids, gallery_ids, gt_vids, gallery_vids, gallery_vids1
    #return gt_ids, gallery_ids, gt_vids, gallery_vids
        

def long_lat(gallery_ids):
    
    #root = '/home/c3-0/shruti/data/bdd_vgl'
    # load prediction lat and long
    video_name1 = (gallery_ids).split("_")[0]
    id1 = int((gallery_ids).split("_")[1]) # second corresponds to timestamp
    meta_data_path = os.path.join(opt.data_folder, 'info', 'val', video_name1+'.json')
    meta_data = json.load(open(meta_data_path, 'r'))
    #print(id1)
    t0 = meta_data['startTime'] # starting time of video recording
    timestamp_id = int((id1*1000 + t0)/1000)
    lat = 0
    lon = 0
    gps_seq = meta_data['locations']
    for ts in gps_seq:
        ts_1 = ts['timestamp']
        if int(ts_1/1000) == timestamp_id:
            lon = ts['longitude']
            lat = ts['latitude']

    return lon, lat
        

def distances(pid, gid):
    
    global dist_dict, lat_lon_map, cnt

    i = None
    j = None
    if pid in lat_lon_map:
        i = lat_lon_map[pid]
        # print('key found!', pid)
    else:
        print('key not found!', pid)
        
    if gid in lat_lon_map:
        j = lat_lon_map[gid]
        # print('key found!', gid)
    else:
        print('key not found!', gid)

    if i and j:
        dist = dist_dict[i,j]
    else:
        cnt += 1
        print('Key not found!', cnt)
        # print(pid, gid)
        dist = distances_1(pid, gid)

    return dist

def distances_1(gallery_ids, gt_id):

    # load prediction lat and long
    src_lon, src_lat = long_lat(gallery_ids)
    
    # load gt lat and long
        
    dst_lon, dst_lat = long_lat(gt_id)
        
    # distances
    y = haversine(src_lon, src_lat, dst_lon, dst_lat) #distance from gt
    
    return y

def find_top_k(preds, gt_id, gallery_ids, gt_vid, gallery_vids, gallery_vids1, opt):
    
    top_k = np.zeros(4) # [0, 0, 0, 0] # top-1, top-5, top-10, top-1%
    #top_k = np.zeros(6) # [0, 0, 0, 0, 0, 0] # top-1, top-5, top-10, top-1%, top-2%, top-100
    FLAG = True # to avoid unncessary distance cal e.g if added to top-1 automatically add to top-5 and so on.

    print("ground truth Id:", gt_id) # # e.g. b1c66a42-6f7d68ca_1 
    #print("top-prediction-id", preds[0])
    print("top-prediction", gallery_ids[preds[0]]) # # e.g. b1c66a42-6f7d68ca_1 
    if gt_id == gallery_ids[preds[0]]:
        top_k[0] += 1 # top-1
        print("CORRECT TOP PREDICTION")
           
    else:
                
        # distances
        #y = haversine(src_lon, src_lat, dst_lon, dst_lat) #distance from gt
        y = distances(gallery_ids[preds[0]], gt_id)
        
        print("top prediction dist from ground truth", y)
        if y <= 0.05:
            top_k[0] += 1 # top-1 within a limit of 0.05 mile
            print("added to correct top-1")
            FLAG = False
            

    pred_ids = [gallery_ids[i] for i in preds[:5]]
    if gt_id in pred_ids:
        top_k[1] += 1 # top-5
        print("in top-5")
        
    else:
        y1 = []
        y1.append(y)
        if FLAG:
            for i in range(1, 5):
                y1.append(distances(gallery_ids[preds[i]], gt_id))
            if min(y1) <= 0.05:
                top_k[1] += 1 # top-5 within a limit of 0.05 mile
                print("min dist from ground truth", min(y1))
                print("added to correct top-5")
                FLAG = False
        else:
            top_k[1] += 1 # top-5 within a limit of 0.05 mile
            print("added to correct top-5 for top-1")
            
    
    pred_ids = [gallery_ids[i] for i in preds[:10]]
    if gt_id in pred_ids:
        top_k[2] += 1 # top-10
        print("in top-10")
    else:
        
        if FLAG:
            for i in range(5, 10):
                y1.append(distances(gallery_ids[preds[i]], gt_id))
            if min(y1) <= 0.05:
                top_k[2] += 1 # top-10 within a limit of 0.05 mile
                print("added to correct top-10")
                print("min dist from ground truth", min(y1))
                FLAG = False
        else:
            top_k[2] += 1 # top-10 within a limit of 0.05 mile
            print("added to correct top-10 for top-1 or top-5")

    #pred_ids = [gallery_ids[i] for i in preds[:100]]
    #if gt_id in pred_ids:
    
    #    top_k[5] += 1 # top-100
    
        
    # print(pred_ids, gt_id)
    if _DEBUG:
        save_results(gt_id, gt_id, preds[:5], gallery_vids, gallery_ids, gallery_vids1, opt)

    one_per = int(len(gallery_ids)/100.) # top-1%
    #two_per = 2 * int(len(gallery_ids)/100.) # top-2%
    #ten_per = 5 * int(len(gallery_ids)/100.) # top-5%

    pred_ids = [gallery_ids[i] for i in preds[:one_per]]
    if gt_id in pred_ids:
        top_k[3] += 1 # top-1%
        print("in top-1%")
    else:
        
        if FLAG:
            for i in range(10, one_per):
                y1.append(distances(gallery_ids[preds[i]], gt_id))
            if min(y1) <= 0.05:
                top_k[3] += 1 # top-1 within a limit of 1 mile
                print("added to correct top-1%")
                print("min dist from ground truth", min(y1))
                FLAG = False
        else:
            top_k[3] += 1 # top-1 within a limit of 1 mile
            print("added to correct top-1% for top-1, 5 or 10")
                
    #pred_ids = [gallery_ids[i] for i in preds[:two_per]]
    #if gt_id in pred_ids:
    #    top_k[4] += 1 # top-2%

    #for i in range(ten_per):
    
    #    if gt_id == gallery_ids[preds[i]]:
    #        print("location/rank of gt:", i)
            
    
    return top_k 
    # top-1, top-5, top-10, top-1%
    # top-1, top-5, top-10, top-1%, top-2%, top-100


def save_results(gt_vid, gt_id, preds, gallery_vids, gallery_ids, gallery_vids1, opt):
    
    # gt_id and gallery_ids[preds[i]] carry similar value for matching e.g. b6f77fda-46184a8c_5
    # preds[i] e.g. 23973
    
    # g_id e.g. b6f77fda-46184a8c_5
    # gt_vid e.g. b6f77fda-46184a8c_5
    
    s_path = os.path.join('./predictions_initial', str(gt_vid))
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    vid_name = gt_vid.split('_')[0] # gt_vid # e.g. b1c66a42-6f7d68ca_1


    gt_id1 = gallery_vids1[gt_id] #gt_id1 e.g. b1c66a42-6f7d68ca/02.jpeg
    
    
    #if gt_id1 <10:
        #gt_id1 = '0' + gt_id1

    src_gt = os.path.join(opt.data_folder, 'satelite_data', 'val_gps', gt_id1)
    
    dst_gt = os.path.join(s_path, gt_id+'.png')
    copyfile(src_gt, dst_gt)

    pred_ids = [gallery_ids[i] for i in preds]
    

    for i, g_id in enumerate(pred_ids):
             
        g_id2 = gallery_vids1[g_id] # g_id e.g. bc9e1979-f1fb1692_4
        #g_id2 e.g. b1c66a42-6f7d68ca/02.jpeg
        src_img = os.path.join(opt.data_folder, 'satelite_data', 'val_gps', g_id2)
        
        dst_img = os.path.join(s_path, str(i)+'_'+g_id+'.png')
        #print(dst_img)
        copyfile(src_img, dst_img)
        
        


def ret_lat_lon(x_tyle,y_tyle):                                                                                                                          
    # This function returns the lat, lon as a tuple
    # Takes x_tyle, y_tyle and zoom_level
    n = 2.**19
    lon_deg = int(x_tyle)/n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1. - 2. * int(y_tyle)/n)))
    lat_deg = math.degrees(lat_rad)
    # lat_deg = lat_rad * 180.0 / math.pi
    return lat_deg, lon_deg

def haversine_dist(d1, d2):                                                                                                                            
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lon1 = d1
    lat2, lon2 = d2
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # 6371 is Radius of earth in kilometers. Use 3956 for miles
    return c * r

def haversine(lon1, lat1, lon2, lat2):                                                                                                                            
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # 6371 is Radius of earth in kilometers. Use 3956 for miles
    return c * r


def find_dist(src_id, dst_id):

    src_x, src_y = src_id.split('_')
    dst_x, dst_y = dst_id.split('_')

    src_lat, src_lon = ret_lat_lon(src_x, src_y)
    dst_lat, dst_lon = ret_lat_lon(dst_x, dst_y)

    dist = haversine(src_lon, src_lat, dst_lon, dst_lat)

    return dist

def get_scores(inds, gt_ids, gallery_ids, gt_vids, gallery_vids, gallery_vids1, opt):

    top_k = np.zeros(4) # [0, 0, 0, 0] # # top-1, top-5, top-10, top-1%

    num_samples, _ = inds.shape

    for i in range(num_samples):
        preds = inds[i]
        # print('{} {} {}'.format(gt_ids[i], preds[0], gallery_ids[preds[0]]))
        c_top_k = find_top_k(preds, gt_ids[i], gallery_ids, gt_vids[i], gallery_vids, gallery_vids1, opt)
        top_k += c_top_k

        # c_dist = find_dist(gallery_ids[preds[0]], gt_ids[i])
        c_dist = distances(gallery_ids[preds[0]], gt_ids[i])
        dist += c_dist

    return top_k, dist


def get_lat_lon(gt_vids):

    lat_lon_dict = {}

    #root = '/home/c3-0/shruti/data/bdd_vgl'

    for vid in gt_vids:
        
        meta_data_path = os.path.join(opt.data_folder, 'info', 'val', vid+'.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        #print(id1)
        t0 = meta_data['startTime'] # starting time of video recording
        gps_seq = meta_data['locations']
        for i,ts in enumerate(gps_seq):
            ts_1 = ts['timestamp']
            tid = int(round((ts_1-t0)/1000))
            key = str(vid)+'_'+str(tid)
            lon = ts['longitude']
            lat = ts['latitude']
            lat_lon_dict[key] = [lat, lon]

    return lat_lon_dict

def compute_distance(gt_ids, gt_vids):

    #root = '/home/c3-0/shruti/data/bdd_vgl'

    dist_mat = None
    lat_lon_map = None
    # spath_d = './save/dist_mat.npy'
    # spath_m = './save/lat_lon_map.npy'
    spath_d = os.path.join('./list/dict/dist_mat.npy')
    spath_m = os.path.join('./list/dict/lat_lon_map.npy')

    if os.path.exists(spath_d) and os.path.exists(spath_m):
        dist_mat = np.load(open(spath_d, 'rb'), allow_pickle=True)
        lat_lon_map = np.load(open(spath_m, 'rb'), allow_pickle=True)

        return dist_mat, lat_lon_map.item()

    batch_size = 40000
    
    print('computing distances...')

    print('Total samples', len(gt_ids))

    gt_ids_u = list(set(gt_ids))
    print('Total unique video ids', len(gt_ids_u))

    print('Total videos', len(gt_vids))

    # gt_vids_u = list(set(gt_vids))[:10]
    gt_vids_u = list(set(gt_vids))
    # print(gt_vids_u[0])
    print('Total videos', len(gt_vids_u))

    # return dictionary of lat lon
    lat_lon_dict = get_lat_lon(gt_vids_u)
    print('Size of dict:', len(lat_lon_dict))

    lat_lon = list(lat_lon_dict.values())
    print('Total lat lon values:', len(lat_lon))

    # compute distance in batches due to memory limit
    num_samples = len(lat_lon)
    num_batches = int(num_samples/batch_size)
    # num_batches = 1
    dist_mat = []
    for i in range(num_batches):

        # compute pair wise distance between samples
        dist = distance.cdist(lat_lon[i*batch_size:(i+1)*batch_size], lat_lon, haversine_dist)
        # print(dist[0])
        dist_mat.extend(dist)

    if num_batches*batch_size < num_samples:
        # compute pair wise distance between samples
        dist = distance.cdist(lat_lon[num_batches*batch_size:], lat_lon, haversine_dist)
        # print(dist[0])
        dist_mat.extend(dist)

    dist_mat = np.array(dist_mat, dtype='float16')
    print(dist_mat.shape)
    # print(dist_mat[0,0:10])
    lat_lon_map = {k:v for v,k in enumerate(list(lat_lon_dict.keys()))}

    np.save(open(spath_d,'wb'), dist_mat, allow_pickle=True)
    np.save(open(spath_m, 'wb'), lat_lon_map, allow_pickle=True)

    return dist_mat, lat_lon_map

def evaluate(v_emb, g_emb, opt):

    global dist_dict, lat_lon_map

    # number of samples per batch
    chunk = 10000

    # load the gallery and the ground truth
    gt_ids, gallery_ids, gt_vids, gallery_vids, gallery_vids1 = get_ids(opt)
    #gt_ids, gallery_ids, gt_vids, gallery_vids = get_ids(opt)
    #print(len(gt_ids))
    #print(len(gallery_ids))
    #print(len(gallery_vids))
    #print(len(gt_vids))
    # A #test center e.g. 1, 1, b1c66a42-6f7d68ca_1, b1c66a42-6f7d68ca
    # B #test center e.g. b1c66a42-6f7d68ca_1, b1c66a42-6f7d68ca_1, b1c66a42-6f7d68ca_1, b1c66a42-6f7d68ca
    # old e.g. 154379_197218, 154379_197218, b1c66a42-6f7d68ca_1, b1c66a42-6f7d68ca

    #print('computing distances...')
    #t0 = time.time()
    # compute the distance before hand
    dist_dict, lat_lon_map = compute_distance(gt_ids, gt_vids)
    print('dictionary loaded!')
    #print('time:', (time.time()-t0)/60)

    num_samples, dim = v_emb.shape
    
    top_k = np.zeros(4) # [0, 0, 0, 0, 0, 0] # # top-1, top-5, top-10, top-1%, top-2%, top-100
    error = 0.
    #fname = open(fname,'w') # print('whatever', file=fname)

    num_batches = int(num_samples/chunk)
    # iterate for each batch
    for b in range(num_batches):
        dist = distance.cdist(v_emb[b*chunk:(b+1)*chunk,:], g_emb, 'sqeuclidean') # distance between each v_emb: video embedding and g_emb: aerial embedding in array form
        inds = np.argsort(dist, axis=1) # is this axis correct? # indices for sorted array
        print(np.shape(dist))
        #print("gt:", gt_ids[b*chunk:(b+1)*chunk,:])
        print("distance of top pred from gt:", dist[:,0])
        #print('distance of top pred from gt:', file=fname)

        c_top_k, c_error = get_scores(inds, gt_ids[b*chunk:(b+1)*chunk], gallery_ids, gt_vids[b*chunk:(b+1)*chunk], gallery_vids, gallery_vids1, opt)
        #c_top_k, c_error = get_scores(inds, gt_ids[b*chunk:(b+1)*chunk], gallery_ids, gt_vids[b*chunk:(b+1)*chunk], gallery_vids, opt)

        top_k += c_top_k
        error += c_error

    # for left over samples
    if num_batches*chunk < num_samples:
        # test the remaining samples
        dist = distance.cdist(v_emb[num_batches*chunk:,:], g_emb, 'sqeuclidean') # distance 
        inds = np.argsort(dist, axis=1)

        c_top_k, c_error = get_scores(inds, gt_ids[num_batches*chunk:], gallery_ids, gt_vids[num_batches*chunk:], gallery_vids, gallery_vids1, opt)

        top_k += c_top_k
        error += c_error

    top_k = top_k/num_samples
    error = error/num_samples

    return top_k, error


def main():

    t0 = time.time()

    opt = parse_option()

    v_embeddings = None
    g_embeddings = None

    v_embeddings = get_v_embeddings(opt)
    g_embeddings = get_embeddings(opt)

    scores = evaluate(v_embeddings, g_embeddings, opt)

    print(scores)

    t1 = time.time()

    print('time taken:', t1-t0)

if __name__ == '__main__':
    main()
