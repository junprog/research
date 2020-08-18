import torch
import torch.nn as nn
import torchvision.models as models

from . import bagnet33_res18, bagnet_res50

class MyModel(nn.Module):
    def __init__(self, down_scale_num=3, model='ResNet', bag_rf_size=33):
        super(MyModel,self).__init__()

        ## feature_extracter : ResNet, VGG, BagNetの最終fc層なくした事前学習モデル
        ## down_channels : channel数を削減する
        if model == 'ResNet':
            self.feature_extracter = make_resnet18_feature_extracter(down_scale_num)
            self.down_channels = make_resnet18_down_channels(down_scale_num)
        elif model == 'VGG16':
            self.feature_extracter = make_vgg16_feature_extracter(down_scale_num)
            self.down_channels = make_vgg16_down_channels(down_scale_num)
        elif model == 'BagNet':
            self.feature_extracter = make_bagnet_feature_extracter(down_scale_num)
            self.down_channels = make_resnet18_down_channels(down_scale_num)
        elif model == 'BagNet_base50':
            self.feature_extracter = make_bagnet_base50_feature_extracter(down_scale_num, bag_rf_size)
            self.down_channels = make_resnet50_down_channels(down_scale_num)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        ## とりあえずサイズ戻す(upsampleする)

    def forward(self, x):
        x = self.feature_extracter(x)
        x = self.down_channels(x)
        x = self.output_layer(x)

        return x

def make_resnet18_feature_extracter(down_scale_num):
    model = models.resnet18(pretrained=True)

    layers = list(model.children())[:-2]

    extracter = nn.Sequential()
    for i in range(0, down_scale_num):
        if i == 0:
            extracter.add_module('conv2d',layers[0])
            extracter.add_module('bn2d',layers[1])
            extracter.add_module('relu',layers[2])
            extracter.add_module('maxpool',layers[3])
        else:
            extracter.add_module('block{}'.format(i),layers[i+3])

    return extracter

def make_vgg16_feature_extracter(down_scale_num):
    model = models.vgg16(pretrained=True)
    if down_scale_num == 3:
        layers = list(model.features.children())[:-8]
    elif down_scale_num == 4:
        layers = list(model.features.children())[:-1]

    return nn.Sequential(*layers)

def make_bagnet_feature_extracter(down_scale_num):
    model = bagnet33_res18.bagnet33()

    layers = list(model.children())[:-2]

    extracter = nn.Sequential()
    for i in range(0, down_scale_num):
        if i == 0:
            extracter.add_module('conv2d',layers[0])
            extracter.add_module('bn2d',layers[1])
            extracter.add_module('relu',layers[2])
        else:
            extracter.add_module('block{}'.format(i),layers[i+2])

    return extracter

def make_bagnet_base50_feature_extracter(down_scale_num, bag_rf_size):
    if bag_rf_size == 33:
        model = bagnet_res50.bagnet33(pretrained=True)
    elif bag_rf_size == 17:
        model = bagnet_res50.bagnet17(pretrained=True)
    elif bag_rf_size == 9:
        model = bagnet_res50.bagnet9(pretrained=True)

    layers = list(model.children())[:-2]

    extracter = nn.Sequential()
    for i in range(0, down_scale_num):
        if i == 0:
            extracter.add_module('conv1',layers[0])
            extracter.add_module('conv2',layers[1])
            extracter.add_module('bn2d',layers[2])
            extracter.add_module('relu',layers[3])
        else:
            extracter.add_module('block{}'.format(i),layers[i+3])

    return extracter

def make_resnet18_down_channels(down_scale_num):
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

def make_vgg16_down_channels(down_scale_num):
    base_ch = 64
    layers = []

    for i in range(3, 0, -1):
        conv2d = nn.Conv2d(base_ch*(2**(i)), base_ch*(2**(i-1)), kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(base_ch*(2**(i-1))), nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)

def make_resnet50_down_channels(down_scale_num):
    base_ch = 64
    layers = []

    for i in range(down_scale_num, 0, -1):
        #if i == 2:
            #conv2d = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
            #layers += [conv2d, nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True)]
        #else:    
        conv2d = nn.Conv2d(base_ch*(2**(i)), base_ch*(2**(i-1)), kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(base_ch*(2**(i-1))), nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)