"""
import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(pretrained=True)
layers = list(model.children())[:-2]
extracter = nn.Sequential(*layers)

x = torch.rand(1,3,500,500)
out = extracter(x)
print(out)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from mat4py import loadmat

"""
img = cv2.imread('/mnt/hdd02/UCF_CC_50/1.jpg')

data = loadmat('/mnt/hdd02/UCF_CC_50/1_ann.mat')

#for i,location in enumerate(data['image_info']['location']):
for i,location in enumerate(data['annPoints']):
    out = cv2.circle(img, (int(location[0]), int(location[1])), 3, (0,0,255), -1)

cv2.imwrite("img3.png",out)
"""

from scipy.ndimage.filters import gaussian_filter
img_path = '/mnt/hdd02/ShanghaiTech/part_A/train_data/images/IMG_31.jpg'
mat = loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
img= plt.imread(img_path)
k = np.zeros((img.shape[0],img.shape[1]))
#gt = mat["image_info"][0,0][0,0][0]
gt = mat["image_info"]["location"]
for i in range(0,len(gt)):
    if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
        k[int(gt[i][1]),int(gt[i][0])]=1
k = gaussian_filter(k,15)
"""
plt.figure()
plt.imshow(k,interpolation='nearest',vmin=np.min(k), vmax=np.max(k), cmap='jet')
plt.colorbar()
plt.show()
"""
import random
import os
from PIL import Image

train = True

gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
img = Image.open(img_path).convert('RGB')
target = k
if train:
    ratio = 0.5
    crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
    rdn_value = random.random()
    if rdn_value<0.25:
        dx = 0
        dy = 0
    elif rdn_value<0.5:
        dx = int(img.size[0]*ratio)
        dy = 0
    elif rdn_value<0.75:
        dx = 0
        dy = int(img.size[1]*ratio)
    else:
        dx = int(img.size[0]*ratio)
        dy = int(img.size[1]*ratio)

    img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
    target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
    if random.random()>0.8:
        target = np.fliplr(target)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

fig = plt.figure()
a = fig.add_subplot(1,2,1)
b = fig.add_subplot(1,2,2)
a.imshow(img)
b.imshow(target)
plt.show()