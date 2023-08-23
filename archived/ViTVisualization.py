import urllib

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from matplotlib import pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from source.viz.vit_rollout import VITAttentionRollout

if __name__ == '__main__':
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # create the train dataset
    model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    # model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=512).to(device)  # pretrained ViT

    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor.to(device))
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    print(probabilities.shape)

    # show the predicted class
    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


    vit_rollout = VITAttentionRollout(model, device, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.5)
    roll = vit_rollout(1, tensor.to(device))
    # get attention rollout
    layer = 1
    channel = 26
    head = 1

    with torch.no_grad():
        x = model.patch_embed(tensor.to(device))
        x = model._pos_embed(x)
        x = model.norm_pre(x)

        for i in range(layer):  # iterate to before the designated layer
            x = model.blocks[i](x)

        this_attn = model.blocks[layer].attn
        B, N, C = x.shape
        qkv = this_attn.qkv(x).reshape(B, N, 3, this_attn.num_heads, C // this_attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        this_q = q[0, head, 1:, channel]
        this_k = k[0, head, 1:, channel]
        query = torch.sigmoid(this_q).reshape([14, 14]).cpu().detach().numpy()
        key = torch.sigmoid(this_k).reshape([14, 14]).cpu().detach().numpy()

        activation = q[:, head, :, channel] * k[:, head, :, channel].T
        class_activation = activation[0, 1:].reshape([14, 14]).cpu().detach().numpy()

    plt.imshow(key)
    plt.show()

    plt.imshow(query)
    plt.show()

    plt.imshow(img)
    plt.show()

    plt.imshow(class_activation)
    plt.show()

    rollout_image = cv2.resize(roll, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
    plt.imshow(img.resize(size=(400, 400)), alpha=.5)  # plot the original image
    plt.imshow(rollout_image, alpha=.5, cmap='gray')
    plt.show()