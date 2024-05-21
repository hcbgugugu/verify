# -*- coding: utf-8 -*-
# @Time  : 2021/3/20 23:43
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================

from torch.utils.data import Dataset
import os
from skimage import io
import torch
from torchvision import transforms

labels = []
for i in range(10):
    labels.append(48 + i)
for i in range(26):
    labels.append(65 + i)
for i in range(26):###
    labels.append(97 + i)

class VerCodeDataset(Dataset):
    def __init__(self, image_dir="./data/image_train/"):
        l = os.listdir(image_dir)
        self.data = []
        self.label = []
        for d in l:
            fs = os.listdir("{}{}".format(image_dir, d))
            for f in fs:
                fup = "{}{}/{}".format(image_dir, d, f)
                t = torch.from_numpy(io.imread(fup)).float() / 255
                eps = 1e-6  # 一个小的正数  #这两行是我加的
                std = max(t.std(), eps)
                norl = transforms.Normalize(t.mean(), std)

                self.data.append(norl(t.reshape(1, 30, 18)))
                #self.label.append(d)
                #self.label.append(ord(d))
                self.label.append(labels.index(ord(d)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {"data": self.data[item], "label": self.label[item]}

from torch.utils.data import DataLoader

def trainloader(bs):
    ds = VerCodeDataset()
    return DataLoader(ds,shuffle=True,batch_size=bs)

def testloader():
    ds = VerCodeDataset(image_dir="./data/image_test/")
    return DataLoader(ds,batch_size=5)


if (__name__ == "__main__"):
    tl = trainloader(5)
    for step,i in enumerate(tl):
        print(step)
        print(i["data"])
        print(i["label"])
        exit(0)
