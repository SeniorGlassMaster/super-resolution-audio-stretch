import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hyperparameters import *

class Pre_Upscale_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 161, padding=80)
        self.relu1 = nn.Sigmoid()
        self.conv2 = nn.Conv1d(1,1,1)
        self.relu2 = nn.Sigmoid()
        self.conv3 = nn.Conv1d(1,1,91, padding=45)
        
        # super(SRCNN,self).__init__()
        # self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4);
        # self.relu1 = nn.ReLU();
        # self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0);
        # self.relu2 = nn.ReLU();
        # self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2);

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out