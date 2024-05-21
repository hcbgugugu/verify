# -*- coding: utf-8 -*-
# @Time  : 2021/3/23 16:31
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================

from train import Net
import torch
import preprocess as pp
from torchvision import transforms
import data


def preprocess(img_path):
    img_t = pp.get_gray_img("{}".format(img_path))
    img_t_b = pp.binarization(img_t)
    img_t_m = pp.median_filter(img_t_b)
    images, _ = pp.spilt(img_t_m, "____", margin=[5, 5, 1, 1])
    nor_img = []
    for img in images:
        norl = transforms.Normalize(img.mean(), img.std())
        nor_img.append(norl(img.reshape(1, 30, 18)).type(dtype=torch.float32))
    return nor_img


import os

def usage(img_path):
    net = Net()
    sd = torch.load("./model/net.pt")
    net.load_state_dict(sd)
    imgs = preprocess(img_path)
    labels = []
    for img in imgs:
        weight_lab = net(img.reshape(1, 1, 30, 18))
        max_idx = torch.argmax(weight_lab).item()
        labels.append(chr(data.labels[max_idx]))
    os.rename(img_path, "{}/{}.png".format(os.path.dirname(img_path), "".join(labels)))


if (__name__ == "__main__"):
    path = "./data/new/"
    l = os.listdir(path)
    for f in l:
        usage("{}{}".format(path,f))
