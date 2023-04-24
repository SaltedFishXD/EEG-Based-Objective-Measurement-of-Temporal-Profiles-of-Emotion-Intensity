import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

from scipy import io
import os
#os.environ['CUDA_VISIBLE_DEVICES']="0,1"
#os.environ['CUDA_LAUNCH_BLOCKING']='1'



# In[14]:


class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class Log_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.log(x)
        return torch.clamp(y, min=1e-7, max=10000)

class square_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2
        
        
class SCCNet(nn.Module):
    def __init__(self):
        super(SCCNet, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 32, (32, 1))
        self.Bn1 = nn.BatchNorm2d(32)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(32, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20) #12 or 20
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        # self.LogLayer = Log_layer()
        self.classifier = nn.Linear(20*1*27, 2, bias=True) 
        ##60s --> 20*1*635  10s-->11*20*1*102 6s-->20*1*59 5s --> 20*1*49
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        
        #print(x.shape)
        x = x.view(-1, 20*1*27)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x

class SCCNet_1s_DEAP(nn.Module):
    def __init__(self):
        super(SCCNet_1s_DEAP, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 32, (32, 1))
        self.Bn1 = nn.BatchNorm2d(32)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(32, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        # self.LogLayer = Log_layer()
        self.classifier = nn.Linear(20*1*6, 2, bias=True) 
        ##60s --> 20*1*635  10s-->11*20*1*102 6s-->20*1*59 5s --> 20*1*49
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        
        #print(x.shape)
        x = x.view(-1, 20*1*6)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x
class SCCNet3s(nn.Module):
    def __init__(self):
        super(SCCNet3s, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 62, (62, 1))
        self.Bn1 = nn.BatchNorm2d(62)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(62, 20, (1, 20), padding=(0, 10))
        self.Bn2   = nn.BatchNorm2d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 20))
        # self.LogLayer = Log_layer()
        self.classifier = nn.Linear(20*1*27, 2, bias=True) ##22 CHNL : 20*11*635
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        
        #print(x.shape)
        x = x.view(-1, 20*1*27)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x


class SCCNet5s(nn.Module):
    def __init__(self):
        super(SCCNet5s, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 62, (62, 1))
        self.Bn1 = nn.BatchNorm2d(62)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(62, 20, (1, 20), padding=(0, 10))
        self.Bn2   = nn.BatchNorm2d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 20))
        # self.LogLayer = Log_layer()
        self.classifier = nn.Linear(20*1*47, 2, bias=True) ##4s 20*1*79
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        
        #print(x.shape)
        x = x.view(-1, 20*1*47)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x
