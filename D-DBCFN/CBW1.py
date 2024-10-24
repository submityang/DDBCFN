# Torch
" 这个模型是用于测试 "
import copy
from collections import Counter

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
import joblib
import sklearn
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, \
    camel_to_snake
from dcn1 import DeformableConv2d, SPGDeformableConv2d
import pickle


def get_model(name, sps=None, **kwargs,):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        models: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cuda'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']

    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'CBW':
        patch_size = kwargs.setdefault('patch_size', 11)
        center_pixel = True
        model = MODEL(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])  # loss function
    elif name == 'net':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = net(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()  # loss function
    elif name == 'dconv':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = dconv(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()  # loss function
    elif name == 'msdcn':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'msdcn1':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn1(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'msdcn1_se':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn1_se(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'stride':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = strided(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'msdcn1_1':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn1_1(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'msdcn1_2':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn1_2(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'msdcn2':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        model = msdcn2(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = nn.CrossEntropyLoss()
    elif name == 'SLC':
        patch_size = kwargs.setdefault('patch_size', 25)
        center_pixel = True
        with open('segments.csv', 'rb') as file:
            segment = pickle.load(file)
        # segment = kwargs['segment']
        model = SLC(n_bands, n_classes, patch_size=patch_size, segment=segment)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 10)
        criterion = LOSSMAX
        #criterion = nn.CrossEntropyLoss()

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    # model = model.cuda()
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch // 4,
                                                                        verbose=True))
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBW_BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, padding=1, downsample=None, patch_size=11, count=1):
        super(CBW_BasicBlock, self).__init__()
        self.stride = stride
        self.count = count
        self.padding = padding
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.globalAvgPool = nn.AvgPool2d(patch_size, stride=1, padding=0)
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.conv1d_12 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()
        self.relu_ = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        original_out = x
        out = self.globalAvgPool(x)
        out = out.squeeze(3)
        out0 = out.transpose(2, 1)
        out1 = self.conv1d_1(out0)
        out2 = self.conv1d_12(out1)
        out = out1 + out2 + out0
        out = self.bn(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(3).transpose(2, 1)
        out = out * original_out
        return out


class MODEL(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(MODEL, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        kernel_size = 3
        nb_filter = 16

        self.cbw = nn.Sequential(
            CBW_BasicBlock(self.input_channels, self.input_channels, stride=1, padding=1, patch_size=patch_size,
                           count=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, nb_filter * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(nb_filter * 4),
            nn.LeakyReLU(),
        )
        self.maxpool = nn.MaxPool2d((2, 2), 2, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(nb_filter * 4, nb_filter * 8, kernel_size, padding=1),
            nn.BatchNorm2d(nb_filter * 8),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nb_filter * 8, nb_filter * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nb_filter * 16),
            nn.LeakyReLU()
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.BatchNorm1d(n_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.maxpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x = self.cbw(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        out = self.fc_2(x)
        return out


class net(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=1):
        super(net, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        kernel_size = 3
        nb_filter = 16

        self.cbw = nn.Sequential(
            CBW_BasicBlock(self.input_channels, self.input_channels, stride=1, padding=1, patch_size=patch_size,
                           count=1),
        )
        self.conv_5x5 = nn.Conv3d(
            1, 128, (input_channels, 5, 5), stride=(1, 1, 1), padding=0
        )
        self.conv_3x3 = nn.Conv3d(
            1, 128, (input_channels, 3, 3), stride=(1, 1, 1), padding=0
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (input_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(384, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        x = self.cbw(x)
        a = x.size()
        b = []
        for i in range(0, len(a)):
            if i == 0:
                b.append(a[i])
                b.append(1)
            else:
                b.append(a[i])
        b = tuple(b)
        x = x.reshape(b)
        x_5x5 = self.conv_5x5(x)
        x_3x3 = self.conv_3x3(x)
        x_3x3 = x_3x3.tolist()
        for i in range(0, len(x_3x3)):
            for j in range(0, len(x_3x3[i])):
                for k in range(0, len(x_3x3[i][j])):
                    del x_3x3[i][j][k][0]
                    del x_3x3[i][j][k][-1]
                    del x_3x3[i][j][k][0][0]
                    del x_3x3[i][j][k][0][-1]
        x_3x3 = torch.tensor(x_3x3)
        x_1x1 = self.conv_1x1(x)
        x_1x1 = x_1x1.tolist()
        for i in range(0, len(x_1x1)):
            for j in range(0, len(x_1x1[i])):
                for k in range(0, len(x_1x1[i][j])):
                    del x_1x1[i][j][k][0]
                    del x_1x1[i][j][k][0]
                    del x_1x1[i][j][k][-1]
                    del x_1x1[i][j][k][-1]
                    del x_1x1[i][j][k][0][0]
                    del x_1x1[i][j][k][0][0]
                    del x_1x1[i][j][k][0][-1]
                    del x_1x1[i][j][k][0][-1]
        x_1x1 = torch.tensor(x_1x1)
        x = torch.cat([x_5x5, x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)
        d = x.size()
        c = []
        for i in range(0, len(d)):
            c.append(d[i])
        c.append(1)
        c.append(1)
        c = tuple(c)
        x = x.resize(*c)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class dconv(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(dconv, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.dpool = nn.Sequential(
            DeformableConv2d(108, 108, (2, 2), stride=(2, 2), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.dconv = nn.Sequential(
            DeformableConv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.dpool(x)
            x = self.conv5(x)
            x = self.dconv(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dpool(x)
        x = self.conv5(x)
        x = self.dconv(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


class msdcn(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.dpool = nn.Sequential(
            DeformableConv2d(108, 108, (2, 2), stride=(2, 2), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.dconv = nn.Sequential(
            DeformableConv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.se = SELayer(256)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.dpool(x)
            x = self.conv5(x)
            x1 = self.dconv(x)
            x2 = self.conv6(x)
            x = torch.concat((x1, x2), dim=1)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dpool(x)
        x = self.conv5(x)
        x = self.dconv(x)
        x1 = self.dconv(x)
        x2 = self.conv6(x)
        x = torch.concat((x1, x2), dim=1)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class msdcn1(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn1, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x1 = self.conv1_1(x)
            x2 = self.conv1_2(x)
            x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x = torch.concat((x1, x2), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class msdcn1_se(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn1_se, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.se = SELayer(192)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x1 = self.conv1_1(x)
            x2 = self.conv1_2(x)
            x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x_source):
        x = copy.deepcopy(x_source[:, :103, :, :])
        loc = copy.deepcopy(x_source[:, 103, :, :])
        # print(loc)
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x = torch.concat((x1, x2), dim=1)
        x = self.se(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class msdcn1_1(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn1_1, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1_1(x)
            # x2 = self.conv1_2(x)
            # x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x = self.conv1_1(x)
        # x2 = self.conv1_2(x)
        # x = torch.concat((x1, x2), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x



class msdcn1_2(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn1_2, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            # x1 = self.conv1_1(x)
            x = self.conv1_2(x)
            # x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        # x1 = self.conv1_1(x)
        x = self.conv1_2(x)
        # x = torch.concat((x1, x2), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class msdcn2(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(msdcn2, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            DeformableConv2d(64, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 64, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.pool1 = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            DeformableConv2d(256, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv2_2 = nn.Sequential(
            DeformableConv2d(128, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(1024, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x1_1 = self.conv1_1(x)
            x1_2 = self.conv1_2(x)
            x = torch.concat((x1_1, x1_2), dim=1)
            x = self.pool1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x = torch.concat((x2_1, x2_2), dim=1)
            x1 = self.pool2(x)
            x = self.conv3(x1)
            x = x + x1
            x2 = self.conv4(x)
            x = x + x2
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x = torch.concat((x1_1, x1_2), dim=1)
        x = self.pool1(x)
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x = torch.concat((x2_1, x2_2), dim=1)
        x1 = self.pool2(x)
        x = self.conv3(x1)
        x = x + x1
        x2 = self.conv4(x)
        x = x + x2
        x3 = self.conv5(x)
        x = x + x3
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class FPN(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(FPN, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels

    def Pool_block(self, in_channel):
        pool = nn.Sequential(
            nn.Conv2d(in_channel, in_channel/2, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(in_channel/2),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), 2, 1),
            nn.BatchNorm2d(in_channel/2),
            nn.LeakyReLU()
        )
        return pool

    def Dconv_block1(self, in_channel):
        layers1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            DeformableConv2d(32, 64, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        layers2 = nn.Sequential(
            DeformableConv2d(in_channel, 32, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        return [layers1, layers2]

    def Dconv_block(self, in_channel):
        layers1 = nn.Sequential(
            nn.Conv2d(in_channel, 2*in_channel, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(2*in_channel),
            nn.LeakyReLU(),
            DeformableConv2d(2*in_channel, 4*in_channel, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(4*in_channel),
            nn.LeakyReLU()
        )
        layers2 = nn.Sequential(
            DeformableConv2d(in_channel, 2*in_channel, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(2*in_channel),
            nn.LeakyReLU(),
            nn.Conv2d(2*in_channel, 4*in_channel, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(4*in_channel),
            nn.LeakyReLU()
        )
        return [layers1, layers2]

    def Conv3(self, in_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel/2, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(in_channel/2),
            nn.LeakyReLU()
        )
        return layer

    def Up_block(self, in_channel):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x1_1 = self.Dconv_block1(self.input_channels)[0](x)
        x1_2 = self.Dconv_block1(self.input_channels)[1](x)
        x1 = torch.concat((x1_1, x1_2), dim=1)
        x1 = self.Pool_block(128)(x1)
        layer = self.Dconv_block(64)
        x2_1 = layer[0](x1)
        x2_2 = layer[1](x1)
        x2 = torch.concat((x2_1, x2_2), dim=1)
        x2 = self.Pool_block(512)(x2)
        layer = self.Dconv_block(256)
        x3_1 = layer[0](x2)
        x3_2 = layer[1](x2)
        x3 = torch.concat((x3_1, x3_2), dim=1)
        x3 = self.Conv3(2048)(x3)
        return x3


class SLC(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21, segment=None):
        super(SLC, self).__init__()
        self.segment = segment
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.se = SELayer(192)


        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            SPGDeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1, segment=self.segment),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),

            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(128, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x_source = torch.zeros((1, self.input_channels + 1, self.patch_size, self.patch_size,))
            x = copy.deepcopy(x_source[:,0:-1,:,:])
            x1 = self.conv1_1(x)
            x2 = self.conv1_2(x_source)
            x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x_sorce):
        x = copy.deepcopy(x_sorce[:,0:-1,:,:])
        #x=   self.cbw(x)

        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x_sorce)
        x = torch.concat((x1, x2), dim=1)
        x = self.se(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        spa= x.reshape(-1, self.flattened_size)

        x= self.fc_1(spa)

        x = self.fc_2(x)
        return {'spa': spa,
            'x': x,
                'x_sorce':x_sorce}


class strided(nn.Module):
    def initialize_weight(self):
        for m in self.moudles():  # self.moudles是系统里自带的函数{m的表示：依此返回各层}
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
            if isinstance(m, DeformableConv2d):
                init.zeros_(m.weight)
                m.bias += 0.001

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(strided, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.se = SELayer(192)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            DeformableConv2d(96, 96, (3, 3), stride=(1, 1), padding=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            DeformableConv2d(input_channels, 96, (3, 3), stride=(1, 1), padding=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d((2, 2), 2, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # self.se = SELayer(128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(7, padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, n_classes),
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(200, n_classes),
            nn.BatchNorm1d(n_classes)
        )

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x1 = self.conv1_1(x)
            x2 = self.conv1_2(x)
            x = torch.concat((x1, x2), dim=1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.pool(x)
            x = self.conv7(x)
            x = self.avgpool(x)
            t, w, l, b = x.size()
            return t * w * l * b

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x = torch.concat((x1, x2), dim=1)
        x = self.se(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


def train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=100, device=torch.device('cuda'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    # net = nn.DataParallel(net)
    save_epoch = epoch
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        if e == 30:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            # print(data[0][103][0][0])

            optimizer.zero_grad()
            if supervision == 'full':
                a= net(data)
                output = torch.squeeze(a['x'])

                # for i in range(0, len(target)):
                #     target[i] = target[i]
                #     print(target[i])
                # target = target.float()

                #loss = criterion(a,target,net)
                loss = criterion(a, target,net)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                              len(data), len(data) * len(data_loader),
                              100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                          }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                 })
            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e == 100:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e,
                       metric=abs(metric))


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('wk') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        # torch.save(model.state_dict(), model_dir + filename + '.pth')
        torch.save(model, model_dir + filename + '.pth')
    else:
        filename = str('wk')
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.to(torch.float)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            output=output['x']
            if isinstance(output, tuple):
                output = output[0]
            output = torch.squeeze(output)
            output = output.to('cpu')  # 将cpu 改为 cuda

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))

            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out

    return probs


def val(net, data_loader, device='cuda', supervision='full'):
    # TODO : fix me using metrics()p
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total



def contrast_mse(feature, y):
    device = feature.device

    feature_matrix = torch.square(feature.unsqueeze(1) - feature.unsqueeze(0)).sum(-1)
    y_matrix = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
    y_matrix[torch.eye(y_matrix.shape[0], dtype=torch.bool)] = -1


    loss_contrast_pos = (torch.where(y_matrix == 1, torch.tensor(1., device=device), torch.tensor(0., device=device)) * feature_matrix.square()).sum()
    loss_contrast_neg = (torch.where(y_matrix == 0, torch.tensor(1., device=device), torch.tensor(0., device=device)) * F.relu(3.5- feature_matrix).square()).sum()
    loss_contrast = (loss_contrast_pos + loss_contrast_neg) / (2 * (feature.shape[0] ** 2 - feature.shape[0]))

    return loss_contrast

def distance_classifier(x):
        device = x.device
        n = x.size(0)
        m = 14
        d = 14
        anchor = nn.Parameter(torch.eye(14) * 10.0, requires_grad=False).to(device)
        x = x.unsqueeze(1).expand(n, m, d).to(device)
        anchor = anchor.unsqueeze(0).expand(n, m, d).to(device)
        dists = torch.norm(x-anchor, 2, 2).to(device)

        return dists

def conloss( output ,y):
    loss1=nn.CrossEntropyLoss()
    loss3=loss1(output['x'],y)
    loss2=contrast_mse(output['spa'],y)
    loss = loss3

    return  loss

def LOSSMAX(output ,y, model):
    loss1=contrast_mse(output['spa'],y)
    loss2=nn.CrossEntropyLoss()

    distances=distance_classifier(output['x'])
    loss3=CACLoss(distances,y)
    l2_loss = L2Loss(model, 0.001)
    l1_loss = L1Loss(model, 0.001)
    loss = loss1 +loss3['total']-10
    #loss=loss2['total'] -10.0+loss1 +l2_loss+l1_loss

    return loss


def CACLoss( distances, gt):
        device = distances.device
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1).to(device)
        non_gt = torch.Tensor([[i for i in range(14) if gt[x] != i] for x in range(len(distances))]).long().to(device)
        others = torch.gather(distances, 1, non_gt).to(device)

        anchor = torch.mean(true).to(device)

        tuplet = torch.exp(-others + true.unsqueeze(1)).to(device)
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1))).to(device)

        total = 1.5 * anchor + tuplet
        #total =  tuplet
        return {
            'total': total,
            'anchor': anchor,
            'tuplet': tuplet
        }

def L2Loss(model, alpha):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, parma in model.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
        return l2_loss


def L1Loss(model, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
    return l1_loss




class CBW_BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, padding=1, downsample=None, patch_size=11, count=1):
        super(CBW_BasicBlock, self).__init__()
        self.stride = stride
        self.count = count
        self.padding = padding
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.globalAvgPool = nn.AvgPool2d(patch_size, stride=1, padding=0)
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.conv1d_12 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()
        self.relu_ = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        original_out = x
        out = self.globalAvgPool(x)
        out = out.squeeze(3)
        out0 = out.transpose(2, 1)
        out1 = self.conv1d_1(out0)
        out2 = self.conv1d_12(out1)
        out = out1 + out2 + out0
        out = self.bn(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(3).transpose(2, 1)
        out = out * original_out
        return out

def CFA(prediction,device=torch.device('cuda')):
    with open('segments1.csv', 'rb') as file:
        segment = pickle.load(file)
    x, y = segment.shape
    x = generate_sequence(2, x)
    y = generate_sequence(2, y)
    ones_coordinates = []
    lst1 = []
    for i in x:
        for j in y:

            coord = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j), (i, j + 1), (i + 1, j - 1),
                     (i + 1, j), (i + 1, j + 1)]
            lst = [segment[i - 1][j - 1], segment[i - 1][j],
                   segment[i - 1][j + 1], segment[i][j - 1],
                   segment[i][j], segment[i][j + 1],
                   segment[i + 1][j - 1], segment[i + 1][j],
                   segment[i + 1][j + 1]]

            extracted_elements = lst
            counter = Counter(extracted_elements)
            # 选出最多的值
            most_common_num = counter.most_common(1)[0][0]
            for i in range(0, len(lst)):
                if lst[i] == most_common_num:
                    lst1.append(coord[i])
            extracted_elements1 = [prediction[i[0]][i[1]] for i in coord]
            counter1 = Counter(extracted_elements1)
            # 选出最多的值
            most_common_num1 = counter1.most_common(1)[0][0]
            for coord1 in lst1:
                prediction[coord1] = most_common_num1

def generate_sequence(a, n):
    b = int(n / 2 - 1)
    sequence = []
    for i in range(1, b+1):
        sequence.append(a*i - 1)
    return sequence



