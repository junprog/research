import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(pretrained=True)
layers = list(model.children())[:-2]
extracter = nn.Sequential(*layers)

x = torch.rand(1,3,500,500)
out = extracter(x)
print(out)