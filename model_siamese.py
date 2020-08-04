"""
@author: Chen Feng
"""
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
import numpy as np
import math
import scipy.sparse as sp
import torch.nn.functional as F


######################################################################
# Load model
# ---------------------------
def load_network_easy(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last_siamese')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last_siamese')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    # print('pretrained = %s' % net_original.embedding_net.model.features.conv0.weight[0, 0, 0])
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print('network_original = %s' % network.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    if not os.path.exists(os.path.join('./model', name)):
        os.makedirs(os.path.join('./model', name))
    torch.save(network.state_dict(), save_path)


def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    if not os.path.exists(os.path.join('./model', name)):
        os.makedirs(os.path.join('./model', name))
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class ReFineBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512, layer=2):
        super(ReFineBlock, self).__init__()
        add_block = []
        for i in range(layer):
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if dropout:
                add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512):
        super(FcBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751):
        super(ClassBlock, self).__init__()
        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class BN(nn.Module):
    def __init__(self, input_dim=512):
        super(BN, self).__init__()
        bn = []
        bn += [nn.BatchNorm1d(input_dim)]
        bn = nn.Sequential(*bn)
        bn.apply(weights_init_kaiming)
        self.bn = bn

    def forward(self, x):
        x = self.bn(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = Fc_ClassBlock(2048, class_num, dropout=0.5, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x, f = self.classifier(x)
        return x, f


# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x):
        x = self.classifier.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num=751):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = Fc_ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, use_instance=False):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(1024, 2, dropout=0.75, relu=False)
        self.use_instance = use_instance

    def forward(self, x1, x2=None):
        output1, feature1 = self.embedding_net(x1)
        if x2 is None:
            return output1, feature1
        if not self.use_instance:
            output2, feature2 = self.embedding_net(x2)
            feature = (feature1 - feature2).pow(2)
            result = self.classifier.classifier(feature)
            return output1, output2, result, result, result, result, result, result
        else:
            output2, feature2 = self.embedding_net(x2)
            x12 = torch.cat((x1[:, :, :int(x1.size(2)/2)], x2[:, :, int(x2.size(2)/2):]), 2)
            x21 = torch.cat((x2[:, :, :int(x2.size(2)/2)], x1[:, :, int(x1.size(2)/2):]), 2)
            output12, feature12 = self.embedding_net(x12)
            output21, feature21 = self.embedding_net(x21)
            feature = (feature1 - feature2).pow(2)
            feature11_12 = (feature1 - feature12).pow(2)
            feature11_21 = (feature1 - feature21).pow(2)
            feature22_12 = (feature2 - feature12).pow(2)
            feature22_21 = (feature2 - feature21).pow(2)
            feature12_21 = (feature12 - feature21).pow(2)
            result = self.classifier.classifier(feature)
            result11_12 = self.classifier.classifier(feature11_12)
            result11_21 = self.classifier.classifier(feature11_21)
            result22_12 = self.classifier.classifier(feature22_12)
            result22_21 = self.classifier.classifier(feature22_21)
            result12_21 = self.classifier.classifier(feature12_21)
            return output1, output2, \
                   result, result11_12, result11_21, result22_12, result22_21, result12_21

    def get_embedding(self, x):
        return self.embedding_net(x)



