import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parameters import *

class Post_Upscale_Model(nn.Module):

    def __init__(self):
        super(Post_Upscale_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 201, padding=100)
        self.sigm1 = nn.Sigmoid()
        self.conv2 = nn.Conv1d(1,1,91, padding=45)
        self.sigm2 = nn.Sigmoid()
        self.upscale = nn.ConvTranspose1d(1,1,int(WINDOW_SIZE/2)+1)
        # self.upscale = nn.ConvTranspose1d(1,1,2,stride=2)
        self.conv3 = nn.Conv1d(1,1,91, padding=45)
        self.lin1 = nn.Linear(WINDOW_SIZE, WINDOW_SIZE)
        # self.lin2 = nn.Linear(WINDOW_SIZE, WINDOW_SIZE)

    def forward(self, x):
        out = self.conv1(x)
        out = self.sigm1(out)
        out = self.conv2(out)
        out = self.sigm2(out)
        out = self.upscale(out)
        out = self.conv3(out)
        out = self.lin1(out)
        # out = self.lin2(out)
        return out