# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import random
from lr_scheduler import LRScheduler
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image
import copy
import time
import os
from losses import AngleLoss, ArcLoss
from model import ft_net
from random_erasing import RandomErasing
import yaml
import math
from triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from shutil import copyfile
from samplers import RandomIdentitySampler
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='./pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--erasing_p_plus', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0
print('load finish')


data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#======================== 64 ========================================
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((64, 64), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(64, 64), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]



if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]


if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}
train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])

image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])


dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.batchsize,
        sampler=RandomIdentitySampler(
            image_datasets[x],
            opt.batchsize,
            4), num_workers=0, pin_memory=True,) for x in ['train']}

class_names = image_datasets['train'].classes
inputs, classes = next(iter(dataloaders['train']))

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes['train'])
#=======================128 ================================

transform_train_list1 = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((128, 128), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list1 = [
    transforms.Resize(size=(256, 256), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]



if opt.erasing_p > 0:
    transform_train_list1 = transform_train_list1 + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]


if opt.color_jitter:
    transform_train_list1 = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list1

print(transform_train_list1)
data_transforms1 = {
    'train': transforms.Compose(transform_train_list1),
    'val': transforms.Compose(transform_val_list1),
}


image_datasets1 = {}
image_datasets1['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms1['train'])

image_datasets1['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms1['val'])

dataloaders1 = {
    x: torch.utils.data.DataLoader(
        image_datasets1[x],
        batch_size=opt.batchsize,
        sampler=RandomIdentitySampler(
            image_datasets1[x],
            opt.batchsize,
            4), num_workers=0, pin_memory=True,) for x in ['train']}
#
inputs1, classes1 = next(iter(dataloaders1['train']))
dataset_sizes1 = {x: len(image_datasets1[x]) for x in ['train', 'val']}
print(dataset_sizes1['train'])

#=======================256================================

transform_train_list2 = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 256), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list2 = [
    transforms.Resize(size=(256, 256), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]



if opt.erasing_p > 0:
    transform_train_list2 = transform_train_list2 + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]


if opt.color_jitter:
    transform_train_list2 = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list2
#
# if opt.DA:
#     transform_train_list = [ReIDPolicy()] + transform_train_list

print(transform_train_list2)
data_transforms2 = {
    'train': transforms.Compose(transform_train_list2),
    'val': transforms.Compose(transform_val_list2),
}


image_datasets2 = {}
image_datasets2['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms2['train'])

image_datasets2['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms2['val'])

dataloaders2 = {
    x: torch.utils.data.DataLoader(
        image_datasets2[x],
        batch_size=opt.batchsize,
        sampler=RandomIdentitySampler(
            image_datasets2[x],
            opt.batchsize,
            4), num_workers=0, pin_memory=True,) for x in ['train']}
#
use_gpu = torch.cuda.is_available()
since = time.time()
inputs2, classes2 = next(iter(dataloaders2['train']))

dataset_sizes2 = {x: len(image_datasets2[x]) for x in ['train', 'val']}
print(dataset_sizes2['train'])

######################################################################
# Training the model

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

lr_scheduler = LRScheduler(base_lr=2e-2, step=[30,75],
                           factor=0.1, warmup_epoch=10,
                           warmup_begin_lr=2e-4, warmup_mode='linear')

def train_model(model,  criterion, triplet, num_epochs):
    since = time.time()

    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        lr = lr_scheduler.update(epoch)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True)
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            a = []
            for data in dataloaders[phase]:
                #print(data.shape)
                a.append(data)
            print(len(a))
            for data in dataloaders1[phase]:
                a.append(data)
            for data in dataloaders2[phase]:
                a.append(data)
            print(len(a))
            b = [i for i in a]
            random.shuffle(b)
            for data in b:
                # get the inputs
                inputs, labels = data
                print(inputs.shape)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, outputs1 = model(inputs)

                _, preds = torch.max(outputs1.data, 1)
                loss1 = criterion(outputs1, labels)
                loss2 = triplet(outputs, labels)[0]
                loss = loss1 + 1/2 * loss2
                # backward + optimize only if in training phase


                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    ##########

                # statistics

                running_loss += loss.item() * now_batch_size

                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / (dataset_sizes[phase] + dataset_sizes1[phase] + dataset_sizes2[phase])
            epoch_acc = running_corrects / (dataset_sizes[phase] + dataset_sizes1[phase] + dataset_sizes2[phase])

            with open('./model/%s/%s.txt' % (name, name), 'a') as acc_file:
                acc_file.write(
                    'Epoch: %2d, lr: %.8f, loss1: %f, loss2: %f, Precision: %.8f, Loss: %.8f\n' %
                    (epoch, lr, loss1,  loss2, epoch_acc, epoch_loss))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch < 70:
                    if epoch % 2 == 0:
                        save_network(model, opt.name, epoch)
                    draw_curve(epoch)
                else:
                    # if epoch%2 == 0:
                    save_network(model, opt.name, epoch)
                    draw_curve(epoch)
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, opt.name, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    # ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    # ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))




if not opt.resume:
    model = ft_net(len(class_names))
    opt.nclasses = len(class_names)


######################################################################
# Train and evaluate

dir_name = os.path.join('./model', name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('./train_triplet.py', dir_name + '/train_triplet.py')
    copyfile('./model.py', dir_name + '/model.py')
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = CrossEntropyLabelSmooth(num_classes=len(class_names))
triplet = TripletLoss(margin=1.2)

model = train_model(model, criterion, triplet,
                    num_epochs=80)
