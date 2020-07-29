import os
import random
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageOps
        
class Gaussian_Filtering(object):
    def __init__(self, std):
        self.std = std
    
    def __call__(self, gt_nparray):
        gt_density = gaussian_filter(gt_nparray, self.std)

        return Image.fromarray(gt_density)

class Scale(object):
    def __init__(self, crop_scale):
        self.crop_scale = crop_scale

    def __call__(self, image):
        crop_size = (int(image.size[0]*self.crop_scale),int(image.size[1]*self.crop_scale))
    
        return image.resize(crop_size)

class Target_Scale(object):
    def __init__(self, down_scale_num):
        self.down_scale_num = down_scale_num

    def __call__(self, target):
        target = target.resize(size=(target.size[0]//(2**self.down_scale_num), target.size[1]//(2**self.down_scale_num)), resample=Image.BICUBIC)
        target = np.asarray(target)
        target = target*((2**self.down_scale_num)**2)

        return Image.fromarray(target)


class BagNet_Target_Scale(object):
    def __init__(self, down_scale_num):
        if down_scale_num == 5:
            self.down_scale_num = down_scale_num-1
            self.flag = True
        else:    
            self.down_scale_num = down_scale_num
            self.flag = False

    def __call__(self, target):
        if self.flag:
            target = target.resize(size=(target.size[0]//(2**(self.down_scale_num-1)) - 4, target.size[1]//(2**(self.down_scale_num-1)) - 4), resample=Image.BICUBIC)
        else:    
            target = target.resize(size=(target.size[0]//(2**(self.down_scale_num-1)) - 2, target.size[1]//(2**(self.down_scale_num-1)) - 2), resample=Image.BICUBIC)
        
        target = np.asarray(target)
        target = target*((2**self.down_scale_num)**2)

        return Image.fromarray(target)


class Corner_Center_Crop(object):
    def __init__(self, crop_size_h, crop_size_w, crop_position=None):
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, image):
        width = image.size[0]
        height = image.size[1]

        if self.crop_position == 'c':
            th, tw = (self.crop_size_h, self.crop_size_w)
            x1 = int(round((width - tw) / 2.))
            y1 = int(round((height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.crop_size_w
            y2 = self.crop_size_h
        elif self.crop_position == 'tr':
            x1 = width - self.crop_size_w
            y1 = 0
            x2 = width
            y2 = self.crop_size_h
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = height - self.crop_size_h
            x2 = self.crop_size_w
            y2 = height
        elif self.crop_position == 'br':
            x1 = width - self.crop_size_w
            y1 = height - self.crop_size_h
            x2 = width
            y2 = height

        image = image.crop((x1, y1, x2, y2))

        return image

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(0, len(self.crop_positions)-1)]

class Random_Crop(object):
    def __init__(self, crop_size_h, crop_size_w):
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __call__(self, image):
        image = image.crop((self.left_top[0], self.left_top[1], self.left_top[0]+self.crop_size_w, self.left_top[1]+self.crop_size_h))

        return image

    def rc_randomize_parameters(self, image):
        self.left_top = (random.randint(0,image.size[0]-self.crop_size_w), random.randint(0,image.size[1]-self.crop_size_h))