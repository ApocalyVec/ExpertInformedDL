import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class ExtensionModelSubimage(nn.Module):
    def __init__(self, feature_exatractor, num_classes: int, **kwargs):
        super().__init__()
        self.feature_extractor = feature_exatractor
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, img, *args, **kwargs):
        subimage_xs = [self.feature_extractor(x) for x in img['subimages']]
        # concate the subimage features
        x = torch.cat(subimage_xs, dim=1)
        # pass through the classifier
        x = self.resnet.fc(x)