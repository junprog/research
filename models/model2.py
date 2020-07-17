import torch
import torch.nn as nn
import torchvision.models as models

class MyModel2(nn.Module):
    def __init__(self, deconv=False):
        super(MyModel2,self).__init__()

        self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4 = make_encoder()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if deconv:
            self.decoder_out_chs = [256, 128, 64, 64]
            self.decoder = make_decoder_deconv(self.decoder_out_chs)
            self.output_layer = nn.Sequential(*[
                                    nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=3, padding=1)
            ])
        else:
            self.decoder1 = nn.Sequential(upsampledecoder_residual_block(512, 256))
            self.decoder2 = nn.Sequential(upsampledecoder_residual_block(256*2, 128))
            self.decoder3 = nn.Sequential(upsampledecoder_residual_block(128*2, 64))
            self.decoder4 = nn.Sequential(upsampledecoder_residual_block(64*2, 64))

            self.output_layer = nn.Conv2d(64*2, 1, kernel_size=3, padding=1)

        ## とりあえずサイズ戻す(upsampleする)デコーダー


    def forward(self, x):
        x = self.encoder0(x)

        x0 = self.maxpool(x)

        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x_d = self.upsample(x4)
        x_d = self.decoder1(x_d)
        x_d = torch.cat([x_d, x3], dim=1)

        x_d = self.upsample(x_d)
        x_d = self.decoder2(x_d)
        x_d = torch.cat([x_d, x2], dim=1)

        x_d = self.upsample(x_d)
        x_d = self.decoder3(x_d)
        x_d = torch.cat([x_d, x1], dim=1)

        x_d = self.upsample(x_d)
        x_d = self.decoder4(x_d)
        x_d = torch.cat([x_d, x], dim=1)

        x_d = self.upsample(x_d)

        x = self.output_layer(x_d)

        return x

class upsampledecoder_residual_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsampledecoder_residual_block,self).__init__()

        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):       
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

def make_decoder_upsample(out_chs):
    dilation = 2
    in_ch = 512
    layers =[]

    for i, out_ch in enumerate(out_chs):
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

        block0 = layers[0:3]
        block1 = layers[4]
        block2 = layers[5]
        block3 = layers[6]
        block4 = layers[7]

        return nn.Sequential(*block0), block1, block2, block3, block4