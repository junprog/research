import torch
import cv2
import argparse
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

from options import opt_args
from model import MyModel
#from image_transform 
#from target_transform

from torchsummary import summary

def main():
    ### オプション ###
    opts = opt_args()

    #a = torch.rand(1,3,224,224).cuda()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #mat = loadmat('/home/junya/Documents/crowd_counting/research/mat_weight/ilsvrc-bag18-sc-epoch-12.mat')
    #print(mat)

    ### データセット ###



    a = cv2.imread('/mnt/hdd02/ShanghaiTech/part_A/train_data/images/IMG_14.jpg')
    a = torch.from_numpy(np.expand_dims(a.transpose(2,0,1), axis=0)).clone().cuda()
    a = a.float()
    h,w = a.shape[2:4]
    print(h,w)

    model = MyModel(deconv=False)
    model.cuda()
    #model = CANNet(load_weights=True).cuda()

    print(model)

    out = model(a)
    summary(model, (3,192,256))
    #print(model)
    x = out.to('cpu').detach().numpy().copy().squeeze()

    plt.figure()
    plt.imshow(x)
    plt.show()

if __name__ == '__main__':
    main()