import torchvision
from torch import nn
import torch.nn.functional as F


class ResNet(nn.Sequential):
    def __init__(self,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        return F.softmax(x, dim=1)
