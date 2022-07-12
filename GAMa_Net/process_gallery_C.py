from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import random

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_gallery import GalleryResNet
from losses import SupConLoss
from data.dataset_C import set_loader_gallery
import params as params

# np.random.seed(0)
# random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--feat_dim', type=int, default=512)
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
    model = GalleryResNet(name=opt.model, feat_dim=opt.feat_dim)
    # print(model)

    #model = torch.nn.DataParallel(model)

    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['model'], strict=False)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model)
            # model.v_encoder = torch.nn.DataParallel(model.v_encoder)
            # model.i_encoder = torch.nn.DataParallel(model.i_encoder)
        model = model.cuda()
        cudnn.benchmark = True

    return model


def test(test_loader, model, opt):
    # model.train(mode=False)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    embeddings = []
    with torch.no_grad():
        for idx, images in enumerate(test_loader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            model.eval()
            i_features = model(images)
            # print(i_features[0])
            embeddings.extend(i_features.cpu().numpy())
            # embeddings.extend(i_features)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Infer images: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      .format(
                       idx + 1, len(test_loader), batch_time=batch_time,
                       data_time=data_time))
                sys.stdout.flush()

    return np.array(embeddings)



def get_embeddings(opt):
    
    # build data loader
    test_loader = set_loader_gallery(opt)

    # build model
    model = set_model(opt)

    embeddings = test(test_loader, model, opt)
    print(embeddings.shape)

    return embeddings


def main():

    opt = parse_option()

    embeddings = get_embeddings(opt)

    base_path = os.path.split(opt.weights)[0]
    #embeddings_name = os.path.join(base_path, 'embeddings_gps_C_full.npy')
    embeddings_name = os.path.join(base_path, 'embeddings_gps_C.npy')
    #embeddings_name = os.path.join(base_path, 'embeddings_gps.npy')

    if opt.full_gallery:
        embeddings_name = os.path.join(base_path, 'embeddings_F.npy')

    np.save(embeddings_name, embeddings)
    

if __name__ == '__main__':
    main()
