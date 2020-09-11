import math

import scipy.io as io

import torch
import torch.nn as nn
import torch.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(BasicBlock, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]

        out += residual
        out = self.relu(out)

        return out

class BagNet(nn.Module):
    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        #self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

    def _initialize_params(self, mat_weight_path):
        self.params = io.loadmat(mat_weight_path)
        self.cnt = 0

        for m in self.modules():
            if self.cnt >= 0 and self.cnt <= 3:
                if isinstance(m, nn.Conv2d):
                    m.weight = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).permute(3,2,0,1))
                    self.cnt += 1
                elif isinstance(m, nn.BatchNorm2d):   
                    m.weight = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).squeeze())
                    self.cnt += 1
                    m.bias = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).squeeze())
                    self.cnt += 1
                    m.running_mean, m.running_var = torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1][:,0]).squeeze(), torch.from_numpy((self.params['net']['params'][0][0][0][self.cnt][1][:,1] ** 2) - m.eps).squeeze()
                    self.cnt += 1

            elif self.cnt >=4 and self.cnt <= 83:
                if isinstance(m, nn.Conv2d):
                    m.weight = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).permute(3,2,0,1))
                    self.cnt += 1
                elif isinstance(m, nn.BatchNorm2d):   
                    m.weight = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).squeeze())
                    self.cnt += 1
                    m.bias = nn.Parameter(torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1]).squeeze())
                    self.cnt += 1
                    m.running_mean, m.running_var = torch.from_numpy(self.params['net']['params'][0][0][0][self.cnt][1][:,0]).squeeze(), torch.from_numpy((self.params['net']['params'][0][0][0][self.cnt][1][:,1] ** 2) - m.eps).squeeze()
                    self.cnt += 1

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x

def bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(BasicBlock, [2, 2, 2, 2], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        model._initialize_params('mat_weight/bag33_base18_net.mat')

    return model


def bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(BasicBlock, [2, 2, 2, 2], strides=strides, kernel3=[1,1,1,0], **kwargs)
    if pretrained:
        model._initialize_params('mat_weight/bag17_base18_net.mat')

    return model

def bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(BasicBlock, [2, 2, 2, 2], strides=strides, kernel3=[1,1,0,0], **kwargs)

    return model