import torch
import json
import scipy.io as io

import bagnet33_res18
from torchsummary import summary

model = bagnet33_res18.bagnet33().cuda()
print(model)
summary(model, (3,448,448))

def mcn_conv_to_pytorch(params):
    return nn.Parameter(torch.from_numpy(params).permute(3,2,0,1))

def mcn_bn_to_pytorch(params):
    return nn.Parameter(torch.from_numpy(params).squeeze())

def mcn_bn_mean_and_var_to_pytorch(params):
    return nn.Parameter(torch.from_numpy(params[:,0]).squeeze()), nn.Parameter(torch.from_numpy(params[:,1]).squeeze())

params = io.loadmat('mat_weight/bagnet33_params.mat')

for i, _ in enumerate(params['params'][0]):  
    print(params['params'][0][i][0], params['params'][0][i][1].shape)

print(params['params'][0][3][1][:,0])

import torch.nn as nn

print(mcn_conv_to_pytorch(params['params'][0][0][1]).size())

model.conv1.weight = mcn_conv_to_pytorch(params['params'][0][0][1])

model.bn1.weight = mcn_bn_to_pytorch(params['params'][0][1][1])
model.bn1.bias = mcn_bn_to_pytorch(params['params'][0][2][1])
model.bn1.running_mean, model.bn1.running_var = mcn_bn_mean_and_var_to_pytorch(params['params'][0][3][1])

print('\ninsert weight\n')
cnt = 0
for m in model.modules():
    if cnt >= 0 and cnt <= 3:
        if isinstance(m, nn.Conv2d):
            m.weight = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
        elif isinstance(m, nn.BatchNorm2d):   
            m.weight = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
            m.bias = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
            m.running_mean, m.running_var = mcn_bn_mean_and_var_to_pytorch(params['params'][0][cnt][1])
            cnt += 1

    elif cnt >=4 and cnt <= 83:
        if isinstance(m, nn.Conv2d):
            m.weight = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
        elif isinstance(m, nn.BatchNorm2d):   
            m.weight = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
            m.bias = mcn_bn_to_pytorch(params['params'][0][cnt][1])
            cnt += 1
            m.running_mean, m.running_var = mcn_bn_mean_and_var_to_pytorch(params['params'][0][cnt][1])
            cnt += 1

model.cuda()
summary(model, (3,224,224))