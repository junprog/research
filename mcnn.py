import torch
import torch.nn as nn

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN,self).__init__()

        self.global_seq = nn.Sequential()

        self.local_seq = nn.Sequential()

        self.fusion_seq = nn.Sequential()

    def forward(self, x):