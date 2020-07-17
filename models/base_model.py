import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, down_scale_num=3):
        super(MyModel,self).__init__()

        self.encoder = make_resnet18_encoder(down_scale_num)
        ## ResNet, BagNetの最終fc層なくした事前学習モデル エンコーダー

        self.decoder = make_decoder(down_scale_num)
        ## channel数を削減するデコーダー

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        ## とりあえずサイズ戻す(upsampleする)デコーダー

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)

        return x

def make_decoder(down_scale_num):
    base_ch = 64
    layers = []

    for i in range(down_scale_num, 2, -1):
        #if i == 2:
            #conv2d = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
            #layers += [conv2d, nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True)]
        #else:    
        conv2d = nn.Conv2d(base_ch*(2**(i-2)), base_ch*(2**(i-3)), kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(base_ch*(2**(i-3))), nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)

def make_resnet18_encoder(down_scale_num):
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

"""
def make_vgg16_encoder(down_scale_num):
    model = models.vgg16(pretrained=True)

    return model.features
"""