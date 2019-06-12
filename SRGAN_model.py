import torch.nn as nn

class SRGAN_Model(nn.Module):

    def __init__(self):
        super(SRGAN_Model, self).__init__()

    def forward(self, x):
        out = x
        return out