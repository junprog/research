import os
import argparse

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from options import opt_args

from datasets.ShanghaiTech_B import ShanghaiTech_B 
from my_transform import Gaussion_filtering, Scale, Crop
from model import MyModel
from training_tmp import train_epoch
from validation_tmp import val_epoch

from torchsummary import summary

def main():
    ### オプション ###
    opts = opt_args()

    #a = torch.rand(1,3,224,224).cuda()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #mat = loadmat('/home/junya/Documents/crowd_counting/research/mat_weight/ilsvrc-bag18-sc-epoch-12.mat')
    #print(mat)

    scale_method = Scale(opts.crop_scale)

    crop_method = Crop(opts.crop_size_h, opts.crop_size_w, opts.crop_position)

    gaussian_method = Gaussion_filtering(opts.gaussian_std)

    normalize_method = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    ### データセット ###
    if opts.phase == 'train':
        train_set = ShanghaiTech_B(opts.root_path,
                                   opts.ST_part,
                                   opts.train_json,
                                   opts.phase,
                                   scale_method=scale_method,
                                   crop_method=crop_method, 
                                   gaussian_method=gaussian_method,
                                   normalize_method=normalize_method)
        val_set = ShanghaiTech_B(opts.root_path,
                                   opts.ST_part,
                                   opts.val_json,
                                   opts.phase,
                                   scale_method=scale_method,
                                   crop_method=crop_method, 
                                   gaussian_method=gaussian_method,
                                   normalize_method=normalize_method)
    else:
        #test_set = ShanghaiTech_B(opts.root_path, opts.ST_part, opts.train_json, opts.phase, im_transforms)
        pass

    train_loader = torch.utils.data.DataLoader(train_set,   
                                            shuffle=True,
                                            num_workers=opts.num_workers,
                                            batch_size=opts.batch_size
                                            )

    val_loader = torch.utils.data.DataLoader(val_set,   
                                            shuffle=False,
                                            num_workers=opts.num_workers,
                                            batch_size=opts.batch_size
                                            )

    ### モデル生成 ###
    model = MyModel(deconv=False)
    
    checkpoint = torch.load('/home/junya/Documents/crowd_counting/research/saved_model/save_90.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    print(model)
    summary(model, (3,192,256))

    ## 損失関数,オプティマイザ ##
    criterion = nn.MSELoss(reduction='sum').cuda()

    for epoch in range(opts.start_epoch, opts.num_epochs):
        train_epoch(epoch, train_loader, model, criterion)
        val_epoch(epoch, val_loader, model, criterion)


if __name__ == '__main__':
    main()