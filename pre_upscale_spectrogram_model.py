import torch.nn as nn

class Pre_Upscale_Spectrogram_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Spectrogram_Model, self).__init__()
        self.pad1 = nn.ZeroPad2d((1,1,1,1))
        self.conv1 = nn.Conv2d(1,64,(3,3))
        self.rel1 = nn.ReLU()
        # self.rel1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(64,128,1)
        self.rel2 = nn.ReLU()
        # self.rel2 = nn.Sigmoid()
        self.pad2 = nn.ZeroPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(128,1,(3,3))

    def forward(self, x):
        # print(x.shape)
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.rel1(out)
        out = self.conv2(out)
        out = self.rel2(out)
        out = self.pad2(out)
        out = self.conv3(out)
        return out