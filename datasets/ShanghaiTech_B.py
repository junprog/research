### shanghaitsch partBのデータ（image）のパスから,imageとtargetを返す
import torch
import torch.utils.data as data
from torchvision import transforms

import os
import json
import numpy as np
from PIL import Image
import scipy.io as io

### pathの画像を読み込み、画像データを返す関数 ###
def image_loader(image_path):
    with open(image_path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

### image pathをground truth pathに変換し、ground truthをnumpy配列で返す関数 ###
def target_loader(image_path):
    mat_path = image_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')
    with open(mat_path, 'rb') as f:
        return io.loadmat(f)['image_info'][0,0][0,0][0]

### jsonファイルからリストを返す関数 ###
def json_loader(json_path):
    with open(json_path, 'r') as json_data:
        return json.load(json_data)

## location -> imageサイズの空配列にに1でマッピング
def gt_mapping(image, location):
    zeropad = np.zeros((image.size[0],image.size[1]))

    for i in range(0,len(location)):
        if int(location[i][0]) < image.size[0] and int(location[i][1]) < image.size[1]:
            zeropad[int(location[i][0]),int(location[i][1])] = 1
    zeropad = zeropad.T

    return zeropad

class ShanghaiTech_B(data.Dataset):
    def __init__(self,
                 opts,
                 json_file_name, 
                 scale_method=None,
                 target_scale_method=None,
                 crop_method=None, 
                 gaussian_method=None,
                 normalize_method=None):

        self.phase = opts.phase
        self.model = opts.model
        self.crop_size_w = opts.crop_size_w
        self.crop_size_h = opts.crop_size_h

        self.json_path = os.path.join(opts.dataset, json_file_name)

        self.scale_transform = scale_method
        self.target_scale_tansform = target_scale_method
        self.crop_transform = crop_method
        self.gaussian_transform = gaussian_method
        self.normalize_transform = normalize_method

        self.image_loader = image_loader
        self.target_loader = target_loader
        self.gt_mapping = gt_mapping

        self.image_path_list = json_loader(self.json_path)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image = self.image_loader(self.image_path_list[index])
        target = self.target_loader(self.image_path_list[index])

        if self.phase == 'train':
            if self.crop_transform is not None:
                self.crop_transform.rc_randomize_parameters(image)
                
            self.target_scale_tansform.calc_scale_w(self.crop_size_w)
            self.target_scale_tansform.calc_scale_h(self.crop_size_h)

            target_transforms = transforms.Compose([
                self.gaussian_transform,
                self.crop_transform,
                self.target_scale_tansform,
                transforms.ToTensor()
            ])

            image_transforms = transforms.Compose([
                self.crop_transform,
                transforms.ToTensor(),
                self.normalize_transform
            ])

            num = len(target)
            target = self.gt_mapping(image, target)
            tensor_target = target_transforms(target)
            tensor_image = image_transforms(image)

        elif self.phase == 'test':
            w, h = image.size
            self.target_scale_tansform.calc_scale_w(w)
            self.target_scale_tansform.calc_scale_h(h)

            target_transforms = transforms.Compose([
                self.gaussian_transform,
                self.target_scale_tansform,
                transforms.ToTensor()
            ])

            image_transforms = transforms.Compose([
                transforms.ToTensor(),
                self.normalize_transform
            ])

            num = len(target)
            target = self.gt_mapping(image, target)
            tensor_target = target_transforms(target)
            tensor_image = image_transforms(image)

        return tensor_image, tensor_target, num
    