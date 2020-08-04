# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model_siamese import ft_net, ft_net_dense
from model_siamese import SiameseNet
from random_erasing import RandomErasing
from datasets import SiameseDataset
import yaml
from model_siamese import save_network, save_whole_network

version = torch.__version__
print(version)
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='data_aug', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='market', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_false', help='use densenet121')
parser.add_argument('--use_instance', action='store_true', help='use use_instance')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')

opt = parser.parse_args()
opt.use_dense = True
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
data_dir = os.path.join('data', opt.data_dir, 'pytorch')
print('data_dir = %s' % data_dir)

name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)


######################################################################
# Load Data
# --------------------------------------------------------------------

transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
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

dataset = SiameseDataset
image_datasets['train'] = dataset(os.path.join(data_dir, 'train_all'),
                                  data_transforms['train'])
image_datasets['val'] = dataset(os.path.join(data_dir, 'val'),
                                data_transforms['val'])

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes = %s' % dataset_sizes)

use_gpu = torch.cuda.is_available()

since = time.time()
print(time.time() - since)

######################################################################
# Training the model
# --------------------------------------------------------------------

def train_model_siamese(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_id_loss = 0.0
            running_verif_loss = 0.0
            running_id_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, vf_labels, id_labels = data
                now_batch_size, c, h, w = inputs[0].shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if type(inputs) not in (tuple, list):
                    inputs = (inputs,)
                if type(vf_labels) not in (tuple, list):
                    vf_labels = (vf_labels,)
                if type(id_labels) not in (tuple, list):
                    id_labels = (id_labels,)
                if use_gpu:
                    inputs = tuple(d.cuda() for d in inputs)
                    vf_labels = tuple(d.cuda() for d in vf_labels)
                    id_labels = tuple(d.cuda() for d in id_labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                output1, output2, \
                result, result11_12, result11_21, result22_12, result22_21, result12_21 = model(inputs[0], inputs[1])
                _, id_preds1 = torch.max(output1.detach(), 1)
                _, id_preds2 = torch.max(output2.detach(), 1)
                _, vf_preds = torch.max(result.detach(), 1)
                _, vf_preds11_12 = torch.max(result11_12.detach(), 1)
                _, vf_preds11_21 = torch.max(result11_21.detach(), 1)
                _, vf_preds22_12 = torch.max(result22_12.detach(), 1)
                _, vf_preds22_21 = torch.max(result22_21.detach(), 1)
                _, vf_preds12_21 = torch.max(result12_21.detach(), 1)
                loss_id1 = criterion(output1, id_labels[0])
                loss_id2 = criterion(output2, id_labels[1])
                loss_id = (loss_id1 + loss_id2) / 2.0
                loss_verif0 = criterion(result, vf_labels[0])
                loss_verif1 = criterion(result11_12, vf_labels[1])
                loss_verif2 = criterion(result11_21, vf_labels[2])
                loss_verif3 = criterion(result22_12, vf_labels[3])
                loss_verif4 = criterion(result22_21, vf_labels[4])
                loss_verif5 = criterion(result12_21, vf_labels[5])

                # r1 = 0.85
                r1 = 0.621
                loss_verif = (loss_verif0 + loss_verif1 + loss_verif2 + loss_verif3 + loss_verif4 + loss_verif5) / 6.0
                loss = r1 * loss_id + (1.0 - r1) * loss_verif


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_id_loss += loss_id.item()  # * opt.batchsize
                running_verif_loss += loss_verif.item()  # * opt.batchsize

                running_id_corrects += float(torch.sum(id_preds1 == id_labels[0].detach()))
                running_id_corrects += float(torch.sum(id_preds2 == id_labels[1].detach()))
                running_verif_corrects += float(torch.sum(vf_preds == vf_labels[0]))
                running_verif_corrects += float(torch.sum(vf_preds11_12 == vf_labels[1]))
                running_verif_corrects += float(torch.sum(vf_preds11_21 == vf_labels[2]))
                running_verif_corrects += float(torch.sum(vf_preds22_12 == vf_labels[3]))
                running_verif_corrects += float(torch.sum(vf_preds22_21 == vf_labels[4]))
                running_verif_corrects += float(torch.sum(vf_preds12_21 == vf_labels[5]))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_id_loss = running_id_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_id_acc = running_id_corrects / (datasize * 2)
            epoch_verif_acc = running_verif_corrects / (datasize * 6)
            # epoch_verif_acc = running_verif_corrects / datasize

            print(
                '{} Loss_id: {:.4f}  Loss_verify: {:.4f}  Acc_id: {:.4f}  Acc_verify: {:.4f} '.format(
                    phase, epoch_id_loss, epoch_verif_loss, epoch_id_acc, epoch_verif_acc))

            epoch_acc = (epoch_id_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_id_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, name, 'best_siamese')
                save_network(model, name, 'best_siamese_' + str(opt.save_model_name))
                save_whole_network(model, name, 'whole_best_siamese')

            if epoch % 10 == 9:
                save_network(model, name, epoch)

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    save_network(model, name, 'last_siamese')
    save_network(model, name, 'last_siamese_' + str(opt.save_model_name))
    save_whole_network(model, name, 'whole_last_siamese')
    return model



print('class_num = %d' % len(class_names))
embedding_net = ft_net_dense(len(class_names))
model_siamese = SiameseNet(embedding_net, use_instance=opt.use_instance)
if use_gpu:
    model_siamese.cuda()
print('model_siamese structure')
# print(model_siamese)
criterion = nn.CrossEntropyLoss()

stage_1_classifier_id = list(map(id, model_siamese.embedding_net.classifier.parameters())) \
                        + list(map(id, model_siamese.embedding_net.model.fc.parameters()))
stage_1_verify_id = list(map(id, model_siamese.classifier.parameters()))
stage_1_classifier_params = filter(lambda p: id(p) in stage_1_classifier_id, model_siamese.parameters())
stage_1_verify_params = filter(lambda p: id(p) in stage_1_verify_id, model_siamese.parameters())
stage_1_base_params = filter(lambda p: id(p) not in stage_1_classifier_id + stage_1_verify_id,
                             model_siamese.parameters())

optimizer_ft = optim.SGD([
    {'params': stage_1_base_params, 'lr': 0.1 * opt.lr},
    {'params': stage_1_classifier_params, 'lr': 1 * opt.lr},
    {'params': stage_1_verify_params, 'lr': 1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# If you want to obtain better results, you can increase 'epoc' appropriately.
epoc = 30
step = 9
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[12, 20, 26], gamma=0.1)
print('net_loss_model = %s   epoc = %3d   step = %3d' % (opt.net_loss_model, epoc, step))
model = train_model_siamese(model_siamese, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoc)

