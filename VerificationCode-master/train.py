# -*- coding: utf-8 -*-
# @Time  : 2021/3/21 1:45
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dropout=0.1):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 25, 5)
        self.fc1 = nn.Linear(1 * 25 * 4 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        # 池化出来大小直接除2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


import torch.optim as optim
import data
import torch

import datetime

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)

    opt = optim.SGD(net.parameters(), lr=0.01)
    epoch = 2000
    batch_size = 50
    trainloader = data.trainloader(batch_size)
    st = datetime.datetime.now()
    print(st)
    for e in range(epoch):
        for step, d in enumerate(trainloader):
            data_cuda =  d["data"].to(device)
            label_cuda = d["label"].to(device)
            opt.zero_grad()
            out = net(data_cuda)
            lf = nn.CrossEntropyLoss()
            loss = lf(out, label_cuda)
            loss.backward()
            opt.step()
            if (e % 500 == 0 and step == 1):
                print("e : {} , step : {}, loss : {}".format(e, step, loss))
    print(datetime.datetime.now() - st)
    torch.save(net.state_dict(),"./model/net.pt")
    torch.save(opt.state_dict(), "./model/opt.pt")




if (__name__ == "__main__"):
    train()
