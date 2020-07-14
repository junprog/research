import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, deconv=False):
        super(MyModel,self).__init__()

        self.encoder = make_encoder()
        ## ResNet, BagNetの最終fc層なくした事前学習モデル 

        if deconv:
            self.decoder_out_chs = [256, 128, 64, 64]
            self.decoder = make_decoder_deconv(self.decoder_out_chs)
            self.output_layer = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1, stride=2, output_padding=1)
        else:
            self.decoder_out_chs = ['Up', 256, 'Up', 128, 'Up', 64, 'Up' ,64, 'Up']
            self.decoder = make_decoder_upsample(self.decoder_out_chs)
            self.output_layer = nn.Conv2d(64,1,kernel_size=3,padding=1)
        ## とりあえずサイズ戻す(upsampleする)デコーダー

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        
        return x

def make_decoder_upsample(out_chs):
    dilation = 2
    in_ch = 512
    layers =[]

    for out_ch in out_chs:
        if type(out_ch) is str:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            layers += [upsample]
        else: 
            conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
            layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
    
    return nn.Sequential(*layers)

def make_decoder_deconv(out_chs):
    dilation = 2
    in_ch = 512
    layers =[]

    for out_ch in out_chs:
        conv2d = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1, stride=dilation, output_padding=1)
        layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch

    return nn.Sequential(*layers)

def make_encoder():
        model = models.resnet18(pretrained=True)
        layers = list(model.children())[:-2]

        return nn.Sequential(*layers)
