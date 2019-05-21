import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Model(nn.Module):

    def __init__(self):
        super(NN_Model, self).__init__()
        self.conv1 = nn.Conv2d()