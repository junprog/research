import torch
import json
import scipy.io as io

from models import bagnet33_res18
from torchsummary import summary

model = bagnet33_res18.bagnet33()
#print(model)
#summary(model, (3,224,224))

params = io.loadmat('mat_weight/bagnet33_params.mat')
for i, _ in enumerate(params['params'][0]):  
    print(params['params'][0][i][0], params['params'][0][i][1].shape)

import torch.nn as nn

model.conv1.weight = nn.Parameter(torch.from_numpy(params['params'][0][0][1]))

model.bn1.weight = nn.Parameter(torch.from_numpy(params['params'][0][1][1]))
model.bn1.bias = nn.Parameter(torch.from_numpy(params['params'][0][2][1]))
model.bn1.momentum = nn.Parameter(torch.from_numpy(params['params'][0][3][1]))

print(model.state_dict().keys())

"""
for m in model.modules():
    if isinstance(m, nn.Conv2d):

    elif isinstance(m, nn.BatchNorm2d):

"""        
