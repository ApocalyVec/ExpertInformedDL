import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class ExtensionModelSubimage(nn.Module):
    def __init__(self, feature_exatractor, num_classes: int, **kwargs):
        super().__init__()
        self.feature_extractor = feature_exatractor
        self.fc_combine = nn.Linear(3168 * 6, 1024)
        # Classifier
        self.fc_classifier = nn.Linear(1024, num_classes)
    def forward(self, img, *args, **kwargs):
        subimage_features = [self.feature_extractor(x) for x in img['subimages']]
        pooled_features = []
        for features in subimage_features:
            # Pool and flatten the feature maps
            pooled = [F.adaptive_avg_pool2d(fmap, (1, 1)).view(fmap.size(0), -1) for fmap in features]
            # Concatenate the flattened feature maps
            pooled_features.append(torch.cat(pooled, dim=1))
        # Concatenate features from all subimages
        x = torch.cat(pooled_features, dim=1)
        # Combine features from all subimages
        x = self.fc_combine(x)
        x = F.relu(x)
        # Pass through the classifier
        x = self.fc_classifier(x)
        return x
