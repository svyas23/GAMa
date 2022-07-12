from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from data.dataset import set_loader
import params

scaler = torch.cuda.amp.GradScaler()

torch.backends.cudnn.benchmark = True    

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=8e-5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150,200',
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

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='2',
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
    opt.model_path = './screening_net/save'.format(opt.dataset)
    opt.tb_path = './screening_net/save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_dim{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.feat_dim)

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
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, bsz=opt.batch_size)
    criterion = SupConLoss(temperature=opt.temp)
    
    # enable multiple GPUs
    # model = torch.nn.DataParallel(model)

    #checkpoint = torch.load(opt.weights)
    #model.load_state_dict(checkpoint['model'], strict=False)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            # model.v_encoder = torch.nn.DataParallel(model.v_encoder)
            # model.i_encoder = torch.nn.DataParallel(model.i_encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    # for idx, (videos, images, labels_an, labels_hn) in enumerate(train_loader):
    #for idx, (videos, images) in enumerate(train_loader): # hn
    for idx, (videos, images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            videos = videos.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True) #hn
            
            
        # add hard negatives to batch size
        #videos_p, videos_n = torch.split(videos, [1,1], dim=1) #hn
        #images_p, images_n = torch.split(images, [1,1], dim=1) #hn
        # labels_p, labels_n = torch.split(labels, [1,1], dim=1)
        
        # print(videos.shape)
        # simages_list = [videos[:,i] for i in range(videos.shape[1])]
        simages_list = torch.split(videos, 1, dim=2)
        # print(simages_list[1].shape)
        simages = torch.squeeze(torch.cat(simages_list, dim=0), 2)
        # print(simages.shape)

        # print(videos_p.shape)
        #videos = torch.squeeze(torch.cat([videos_p, videos_n], dim=0), 1) #hn
        #images = torch.squeeze(torch.cat([images_p, images_n], dim=0), 1) #hn
        # labels = torch.squeeze(torch.cat([labels_p, labels_n], dim=0), 1)

        # print(videos.shape)
        # labels = torch.cat([labels_an, labels_hn], dim=0)
        
        # bsz = labels.shape[0]
        bsz = videos.shape[0]
        #bsz = videos.shape[0] * labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            # compute loss
            sq_features, A_features = model(simages, images) # sq_features of sequence of smaller aerial images
            # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # get the images back together
            # img_features = torch.split(sq_features, bsz, dim=0)
            # features = torch.cat()

            features = torch.cat([sq_features.unsqueeze(1), A_features.unsqueeze(1)], dim=1)
            labels = None
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        # optimizer.step()
        scaler.step(optimizer)

        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # Allow Amp to perform casts as required by the opt_level
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
