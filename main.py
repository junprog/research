import os
import json
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
from my_transform import Gaussian_Filtering, Scale, Corner_Center_Crop, Random_Crop, Target_Scale, BagNet_Target_Scale
from models import base_model, base_residual_model
from training import train_epoch
from validation import val_epoch
from test import test
from utils import Logger

from torchsummary import summary

def main():
    ### オプション ###
    opts = opt_args()

    os.mkdir(opts.results_path)

    with open(os.path.join(opts.results_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opts), opt_file)

    #scale_method = Scale(opts.crop_scale)
    scale_method = None

    if opts.model == 'BagNet':
        target_scale_method = BagNet_Target_Scale(opts.down_scale_num)
    else:   
        target_scale_method = Target_Scale(opts.down_scale_num)

    #crop_method = Corner_Center_Crop(opts.crop_size_h, opts.crop_size_w, opts.crop_position)
    crop_method = Random_Crop(opts.crop_size_h, opts.crop_size_w)

    gaussian_method = Gaussian_Filtering(opts.gaussian_std)

    normalize_method = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    ### データセット,データローダー作成 ###
    if opts.phase == 'train':
        train_set = ShanghaiTech_B(opts.root_path,
                                   opts.ST_part,
                                   opts.train_json,
                                   opts.phase,
                                   scale_method=scale_method,
                                   target_scale_method=target_scale_method,
                                   crop_method=crop_method, 
                                   gaussian_method=gaussian_method,
                                   normalize_method=normalize_method)

        val_set = ShanghaiTech_B(opts.root_path,
                                   opts.ST_part,
                                   opts.val_json,
                                   opts.phase,
                                   scale_method=scale_method,
                                   target_scale_method=target_scale_method,
                                   crop_method=crop_method, 
                                   gaussian_method=gaussian_method,
                                   normalize_method=normalize_method)

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
    elif opts.phase == 'test':
        test_set = ShanghaiTech_B(opts.root_path,
                                  opts.ST_part,
                                  opts.test_json, 
                                  opts.phase,
                                  scale_method=None,
                                  target_scale_method=target_scale_method,
                                  crop_method=None, 
                                  gaussian_method=gaussian_method,
                                  normalize_method=normalize_method)
        
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  shuffle=False,
                                                  num_workers=opts.num_workers,
                                                  batch_size=1
                                                  )

    ### モデル生成 ###
    #model = base_residual_model.create_mymodel(down_scale_num=opts.down_scale_num)
    model = base_model.MyModel(down_scale_num=opts.down_scale_num, model=opts.model)

    ### パラメータ代入 ###
    if opts.load_weight:
        check_points = torch.load(opts.model_path)['state_dict']
        
        from collections import OrderedDict
        new_check_points = OrderedDict()
        for saved_key, saved_value in check_points.items():
            if 'encoder' in saved_key:
                model_key = saved_key.replace('encoder', 'feature_extracter')
                new_check_points[model_key] = saved_value
            elif 'decoder' in saved_key:
                model_key = saved_key.replace('decoder', 'down_channels', )
                new_check_points[model_key] = saved_value
            else:
                new_check_points[saved_key] = saved_value

        model.load_state_dict(new_check_points)

    model.cuda()

    print(model)
    summary(model, (3,448,448))

    ## 損失関数,オプティマイザ ##
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

    if opts.phase == 'train':
        os.mkdir(os.path.join(opts.results_path, 'results'))
        os.mkdir(os.path.join(opts.results_path, 'saved_model'))
        os.mkdir(os.path.join(opts.results_path, 'images'))

        train_logger = Logger(os.path.join(opts.results_path, 'results', 'train.log'), ['epoch', 'loss', 'lr'])
        val_logger = Logger(os.path.join(opts.results_path, 'results', 'val.log'), ['epoch', 'loss'])

        for epoch in range(opts.start_epoch, opts.num_epochs+1):
            train_epoch(epoch, train_loader, model, criterion, optimizer, train_logger, opts, scheduler=scheduler)
            val_epoch(epoch, val_loader, model, criterion, val_logger, opts)

    if opts.phase == 'test':
        os.mkdir(os.path.join(opts.results_path, 'results'))
        os.mkdir(os.path.join(opts.results_path, 'images'))

        test_logger = Logger(os.path.join(opts.results_path, 'results', 'test.log'), ['MAE', 'RMSE'])

        test(test_loader, model, test_logger, opts)


if __name__ == '__main__':
    main()