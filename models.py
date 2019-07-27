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

class Pre_Upscale_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Model, self).__init__()
        self.pad1 = nn.ReflectionPad1d(4)
        self.conv1 = nn.Conv1d(1,64,9)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64,32,1)
        self.rel2 = nn.ReLU()
        self.pad2 = nn.ReflectionPad1d(2)
        self.conv3 = nn.Conv1d(32,1,5)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.rel1(out)
        out = self.conv2(out)
        out = self.rel2(out)
        out = self.pad2(out)
        out = self.conv3(out)
        return out

class Pre_Upscale_Spectrogram_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Spectrogram_Model, self).__init__()
        self.pad1 = nn.ReflectionPad2d((1,1,0,0))
        self.conv1 = nn.Conv2d(2,128,(1,3))
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,64,1)
        self.rel2 = nn.ReLU()
        self.pad2 = nn.ReflectionPad2d((1,1,0,0))
        self.conv3 = nn.Conv2d(64,32,(1,3))
        self.rel3 = nn.ReLU()
        self.pad3 = nn.ReflectionPad2d((1,1,0,0))
        self.conv4 = nn.Conv2d(32,2,(1,3))

    def forward(self, x):
        # print(x.shape)
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.rel1(out)
        out = self.conv2(out)
        out = self.rel2(out)
        out = self.pad2(out)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        return out

class SRGAN_Model(nn.Module):

    def __init__(self):
        super(SRGAN_Model, self).__init__()
        self.lin1 = nn.linear(2,64)
    
    def forward(self, x):
        out = self.lin1(x)
        return out
