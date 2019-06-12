import torch.nn as nn
from parameters import WINDOW_SIZE

class Post_Upscale_Model(nn.Module):

    def __init__(self):
        super(Post_Upscale_Model, self).__init__()
        self.pad1 = nn.ReflectionPad1d(100)
        self.conv1 = nn.Conv1d(1, 1, 201)
        self.sigm1 = nn.Sigmoid()
        self.pad2 = nn.ReflectionPad1d(45)
        self.conv2 = nn.Conv1d(1,1,91)
        self.sigm2 = nn.Sigmoid()
        self.upscale = nn.ConvTranspose1d(1,1,int(WINDOW_SIZE/2)+1)
        # self.upscale = nn.ConvTranspose1d(1,1,2,stride=2)
        self.pad3 = nn.ReflectionPad1d(45)
        self.conv3 = nn.Conv1d(1,1,91)
        # self.lin1 = nn.Linear(WINDOW_SIZE, WINDOW_SIZE)
        # self.lin2 = nn.Linear(WINDOW_SIZE, WINDOW_SIZE)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.sigm1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.sigm2(out)
        out = self.upscale(out)
        out = self.pad3(out)
        out = self.conv3(out)
        # out = self.lin1(out)
        # out = self.lin2(out)
        return out