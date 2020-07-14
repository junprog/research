### shanghaitsch partBのデータ（image）のパスから,imageとtargetを返す
import torch
import torch.utils.data as data

import os
import json
import numpy as np
from PIL import Image
import h5py

### pathの画像を読み込み、画像データを返す関数 ###
def image_loader(image_path):
    with open(image_path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

### image pathをground truth pathに変換し、ground truthをnumpy配列で返す関数 ###
def target_loader(image_path):
    gt_path = image_path.replace('.jpg','.h5').replace('images','ground_truth')
    with h5py.File(gt_path, 'r') as ground_truth:
        return np.asarray(ground_truth['density'])

### jsonファイルからリストを返す関数 ###
def json_loader(json_path):
    with open(json_path, 'r') as json_data:
        return json.load(json_data)

class ShanghaiTech_B(data.Dataset):
    def __init__(self, root_path, ST_part, json_file_name, transform=None, image_transform=None, target_transform=None, json_loader=json_loader):
        self.json_path = os.path.join(root_path, ST_part, json_file_name)

        self.image_transform = image_transform      # リサイズorクリップ
        self.target_transform = target_transform    # ガウシャンフィルタ + リサイズorクリップ
        self.transform = transform      # mean, std 処理 

        self.image_loader = image_loader
        self.target_loader = target_loader

        self.image_path_list = json_loader(self.json_path)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image = self.image_loader(self.image_path_list[index])
        target = self.target_loader(self.image_path_list[index])

        if self.image_transform is not None:
            self.image_transform.randomize_parameters()     # ランダムでクロップ箇所を初期化 
            image = self.image_transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    