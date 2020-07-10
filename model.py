import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

        model = models.resnet18(pretrained=True)
        print(model)
        layers = list(model.children())[:-2]
        encoder = nn.Sequential(*layers)

        self.encoder = encoder
        ## ResNet, BagNetの最終fc層なくした事前学習モデル 

        self.decoder_out_chs = [512, 'Up', 256, 'Up', 128, 'Up' ,128, 'Up', 64, 'Up']
        self.decoder = make_decoder(self.decoder_out_chs)
        ## とりあえずサイズ戻す(upsampleする)デコーダー

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        x = self.output_layer(x)
        
        return x

def make_decoder(out_chs):
    dilation = 2
    in_ch = 512
    layers =[]

    for out_ch in out_chs:
        if type(out_ch) is str:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            layers += [upsample]
        else: 
            conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation = dilation)
            layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch

    return nn.Sequential(*layers)

