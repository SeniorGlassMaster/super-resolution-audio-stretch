import torch.nn as nn

class Pre_Upscale_Spectrogram_Model(nn.Module):

    def __init__(self):
        super(Pre_Upscale_Spectrogram_Model, self).__init__()
        self.pad1 = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(1,64,9)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64,32,1)
        self.rel2 = nn.ReLU()
        self.pad2 = nn.ReflectionPad2d(2)
        self.conv3 = nn.Conv2d(32,1,5)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.rel1(out)
        out = self.conv2(out)
        out = self.rel2(out)
        out = self.pad2(out)
        out = self.conv3(out)
        return out