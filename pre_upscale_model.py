import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parameters import *

class Pre_Upscale_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Model, self).__init__()
        self.pad1 = nn.ReflectionPad1d(25)
        self.conv1 = nn.Conv1d(1,1,51)
        self.sigm1 = nn.Sigmoid()
        self.conv2 = nn.Conv1d(1,1,1)
        self.sigm2 = nn.Sigmoid()
        self.pad2 = nn.ReflectionPad1d(12)
        self.conv3 = nn.Conv1d(1,1,25)
        
        # super(SRCNN,self).__init__()
        # self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4);
        # self.relu1 = nn.ReLU();
        # self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0);
        # self.relu2 = nn.ReLU();
        # self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2);

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.sigm1(out)
        out = self.conv2(out)
        out = self.sigm2(out)
        out = self.pad2(out)
        out = self.conv3(out)
        return out