import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ExtensionModelSubimage(nn.Module):
    def __init__(self, feature_exatractor, num_classes: int, **kwargs):
        super().__init__()
        self.feature_extractor = feature_exatractor
        self.fc_combine = nn.Linear(3168 * 6, 1024)
        # Classifier
        self.fc_classifier = nn.Linear(1024, num_classes)

        self.gradient_hooks = None
        self.subimage_features = None  # save the features for visualization

    def activations_hook(self, grad):
        self.subimage_features.append(grad)

    def forward(self, img, add_grad_hooks=False, *args, **kwargs):
        self.gradient_hooks = []
        self.subimage_features = [self.feature_extractor(x) for x in img['subimages']]

        pooled_features = []
        for feature in self.subimage_features:
            if add_grad_hooks:
                hooks = [fmap.register_hook(self.activations_hook) for fmap in feature]
                self.gradient_hooks.append(hooks)

            # Pool and flatten the feature maps
            pooled = [F.adaptive_avg_pool2d(fmap, (1, 1)).view(fmap.size(0), -1) for fmap in feature]
            # Concatenate the flattened feature maps
            pooled_features.append(torch.cat(pooled, dim=1))
        return self.fc_classifier(F.relu(self.fc_combine(torch.cat(pooled_features, dim=1))))


def get_gradcam(model, image, target, criterion=nn.CrossEntropyLoss):
    output = model(image, add_grad_hooks=True)
    loss = criterion()(output, target)
    loss.backward()

    gradcams_subimages = []
    b = image["subimages"][0].shape[0]
    # Assuming that the forward pass has been done, so `self.features` and `self.gradients` are populated
    for subimage_i, (subimage, gradients, features) in enumerate(zip(image["subimages"], model.subimage_features, model.subimage_features)):
        # Compute the Grad-CAM for the current feature map and gradient
        # Weighted feature maps
        gradcam_aggregated = torch.zeros(b, *subimage.shape[2:], dtype=torch.float32).to(subimage.device)
        for fmap, grad in zip(features, gradients):  # for all the auxiliary heads
            grad_weights = torch.mean(grad, [2, 3], keepdim=True)
            gradcam = torch.relu(torch.sum(grad_weights * fmap, dim=1))
            gradcam = normalize_by_sample(gradcam).detach()
            for batch_i in range(b):
                # upsample to match the original image size
                gradcam_aggregated[batch_i] += torchvision.transforms.Resize(subimage.shape[2:])(gradcam[batch_i, None, :])[0]

        # normalize again because we have summed over all the heads
        gradcam_aggregated = normalize_by_sample(gradcam_aggregated)
        gradcams_subimages.append(gradcam_aggregated.detach().cpu().numpy())
        # plt.imshow(gradcam_aggregated[0].detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(gradcam_aggregated[1].detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
    # Remove hooks after computation
    for hook in model.gradient_hooks:
        [h.remove() for h in hook]

    return gradcams_subimages

def normalize_by_sample(gradcam):
    return (gradcam - gradcam.amin(dim=(1, 2), keepdim=True)) / (
                gradcam.amax(dim=(1, 2), keepdim=True) - gradcam.amin(dim=(1, 2), keepdim=True)) + 1e-8