import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from model import MyModel
from con_model import CANNet
from torchsummary import summary

def main():
    #a = torch.rand(1,3,224,224).cuda()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    a = cv2.imread('/mnt/hdd02/ShanghaiTech/part_A/train_data/images/IMG_1.jpg')
    a = torch.from_numpy(np.expand_dims(a.transpose(2,0,1), axis=0)).clone().cuda()
    a = a.float()

    model = MyModel()
    model.cuda()
    #model = CANNet(load_weights=True)

    out = model(a)
    #summary(model, (3,224,224))
    #print(model)
    x = out.to('cpu').detach().numpy().copy().squeeze()

    plt.figure()
    plt.imshow(x)
    plt.show()

if __name__ == '__main__':
    main()