# -*- coding: utf-8 -*-
# @Time  : 2021/4/8 11:41
# @Author : zhoujiangtao
# @Desc : ==============================================
# 自定义残差神经网络识别验证码数据集
# ======================================================

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from preprocess import show_img
# 使用新的数据集加载方法
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import copy
from torchvision.datasets import ImageFolder
import numpy as np


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)


def get_mean_std(ds):
    loader = DataLoader(ds, batch_size=len(ds))
    d = next(iter(loader))
    mean = d[0].mean()
    std = d[0].std()
    pass


def dataloader():
    transform = transforms.Compose([
        transforms.Grayscale(1),  # 彩色图像转灰度图像num_output_channels默认1
        transforms.ToTensor(),
        transforms.Normalize([0.7826], [0.4125])
    ])

    # 这里默认读出来是[1,3,30,18]，要转换成[30,18]
    ds = {"train": datasets.ImageFolder("./data/image_train", transform=transform),
          "test": ImageFolderWithPaths("./data/image_test", transform=transform)}
    # "test": imagefloder_with_path.ImageFolderWithPaths("./data/image_train", transform=transform)}
    # get_mean_std(ds["train"])
    # f = len(ds["train"].classes)
    loader = {"train": DataLoader(ds["train"], batch_size=4,
                                  shuffle=True),
              "test": DataLoader(ds["test"], batch_size=10)}
    return loader


import time


class BasicBlock(nn.Module):
    expansion = 1

    # downsample用来调整维度的，因为官网的实现中，最后几层的layer的outchannel和上层都不一样
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn.track_running_stats = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.inplanes != self.planes or self.stride != 1):
            identity = nn.Conv2d(self.inplanes, self.planes, kernel_size=1, stride=self.stride, bias=False)
        out += identity(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # 这样的池化不会影响大小
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.block1 = BasicBlock(16, 32)
        self.block2 = BasicBlock(32, 64)
        self.block3 = BasicBlock(64, 128)
        # self.block4 = BasicBlock(128, 256)
        # self.block5 = BasicBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 识别验证码一共32类
        self.fc = nn.Linear(128, 62)#改了

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(loader, conti_train=False):
    epoch = 100
    # 修改全连接层的数量
    model = ResNet()

    if (conti_train):
        sd = torch.load("./model/resnet.pt")
        model.load_state_dict(sd, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    best_accu = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_model = copy.deepcopy(model)

    start = time.time()
    for e in range(epoch):
        model.train()
        epoch_loss = 0.0
        epoch_right = 0.0
        epoch_total = 0
        for i, (data, label) in enumerate(loader["train"]):
            optimizer_ft.zero_grad()
            data = data
            label = label
            out = model(data)
            _, preds = torch.max(out, 1)
            loss = criterion(out, label)

            loss.backward()
            optimizer_ft.step()

            epoch_right += torch.sum(torch.eq(label, preds))
            epoch_loss += loss
            epoch_total += data.shape[0]
            pass
        # exp_lr_scheduler.step()
        epoch_acc = epoch_right / epoch_total
        test_acc = test_on_training(loader, model)
        if (test_acc > best_accu):
            best_accu = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # best_model = copy.deepcopy(model)
        if (e % 5 == 0):
            print("train: accu:{}% ({}/{}),epoch {} loss:{} ,test accu:{}% ".format(round(epoch_acc.item() * 100, 2),
                                                                                   epoch_right, epoch_total, e,
                                                                                   epoch_loss.item(), test_acc))
    print("total time:{}".format(time.time() - start))
    torch.save(best_model_wts, "./model/resnet.pt")
    # torch.save(best_model,"./model/resnet.pkl")


def test_on_training(loader, model):
    right = 0.0
    total = 0.0
    model.eval()
    for i, (data, label, path) in enumerate(loader["test"]):
        out = model(data)
        _, preds = torch.max(out, dim=1)
        right += torch.sum(torch.eq(label, preds))
        total += data.shape[0]
    return round(right.item() * 100 / total, 2)


def test(loader):
    right = 0.0
    total = 0.0
    model = ResNet()
    sd = torch.load("./model/resnet.pt")
    model.load_state_dict(sd, strict=False)
    # model.eval()
    for i, (data, label, path) in enumerate(loader["test"]):
        out = model(data)
        _, preds = torch.max(out, dim=1)
        right_index = torch.eq(label, preds).numpy()
        # 讲tuple转成list
        lp = np.array(list(path))
        wrong_pic = lp[~right_index]
        right += torch.sum(torch.eq(label, preds))
        total += data.shape[0]
    accuracy = round(right.item() * 100 / total, 2)
    print("test: {}%({}/{})".format(accuracy, right, total))
    # print("wrong picture :{}".format(wrong_pic))

def test_on_trainset(loader):
    right = 0.0
    total = 0.0
    model = ResNet()
    sd = torch.load("./model/resnet.pt")
    model.load_state_dict(sd, strict=False)
    # model.eval()
    for i, (data, label) in enumerate(loader["train"]):
        out = model(data)
        _, preds = torch.max(out, dim=1)
        right_index = torch.eq(label, preds).numpy()
        # 讲tuple转成list
        right += torch.sum(torch.eq(label, preds))
        total += data.shape[0]
    accuracy = round(right.item() * 100 / total, 2)
    print("test: {}%({}/{})".format(accuracy, right, total))
    # print("wrong picture :{}".format(wrong_pic))

if (__name__ == "__main__"):
    loader = dataloader()
    # data,label = next(iter(loader["train"]))
    # show_img(data.reshape(30,18))
    # train(loader, conti_train=False)
    test(loader)