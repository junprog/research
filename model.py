import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.encoder = 
        ## ResNet, BagNetの最終fc層なくした事前学習モデル 

        self.decoder =
        ## とりあえずサイズ戻す??

    def forward(self, x):

