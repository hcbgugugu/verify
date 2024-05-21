# -*- coding: utf-8 -*-
# @Time  : 2021/3/21 2:08
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================

import data
from train import Net
import torch

def test():
    net = Net()
    sd = torch.load("./model/net.pt")
    net.load_state_dict(sd)
    dl = data.testloader()
    correct = 0
    total = 0
    for step,d in enumerate(dl):
        out = net(d["data"])
        max_index = torch.argmax(out,keepdim=True,dim=1)
        label = d["label"].reshape(-1,1)
        eq = label.eq(max_index).int()
        correct += eq.sum().item()
        total += dl.batch_size
    print("{}% on {} images test".format(correct*100/total,total))

if(__name__=="__main__"):
    test()