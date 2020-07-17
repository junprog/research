import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, block, down_scale_num=3):
        super(MyModel,self).__init__()

        self.encoder = self._make_resnet18_encoder(down_scale_num)
        ## ResNet, BagNetの最終fc層なくした事前学習モデル エンコーダー

        if down_scale_num == 3:
            self.decoder = self._make_decoder(block, 2, 128, 64)
        else:
            self.decoder = nn.Sequential(self._make_decoder(block, 2, 256, 128),
                                         self._make_decoder(block, 2, 128, 64))
        ## channel数を削減するデコーダー

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        ## とりあえずサイズ戻す(upsampleする)デコーダー

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)

        return x

    def _make_resnet18_encoder(self, down_scale_num):
        model = models.resnet18(pretrained=True)

        layers = list(model.children())[:-2]

        encoder = nn.Sequential()
        for i in range(0, down_scale_num):
            if i == 0:
                encoder.add_module('conv2d',layers[0])
                encoder.add_module('bn2d',layers[1])
                encoder.add_module('relu',layers[2])
                encoder.add_module('maxpool',layers[3])
            else:
                encoder.add_module('block{}'.format(i),layers[i+3])

        return encoder

    def _make_decoder(self, block, blocks, in_ch, out_ch):
        downchannels = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch))

        layers = []
        layers.append(block(in_ch, out_ch, downchannels))
        for i in range(1, blocks):
            layers.append(block(out_ch, out_ch))

        return nn.Sequential(*layers)

class Decode_Block(nn.Module):
    def __init__(self, in_ch, out_ch, downchannels=None):
        super(Decode_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downchannels = downchannels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannels is not None:
            residual = self.downchannels(x)

        out += residual
        out = self.relu(out)

        return out

def create_mymodel(**kwargs):
    model = MyModel(Decode_Block, **kwargs)

    return model

"""
def make_vgg16_encoder(down_scale_num):
    model = models.vgg16(pretrained=True)

    return model.features
"""