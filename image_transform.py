import random
from PIL import Image, ImageOps

class Scale(object):
    def __init__(self, crop_scale):
        self.crop_scale = crop_scale

    def __call__(self, image):
        crop_size = (int(image.size[0]*self.crop_scale),int(image.size[1]*self.crop_scale))
    
        return image.resize(crop_size)

class Crop(object):
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