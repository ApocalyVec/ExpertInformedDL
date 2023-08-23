import os.path

import numpy as np
import timm
import torch
from matplotlib import pyplot as plt
from timm.data import resolve_data_config, create_transform

from torch import nn
from torchsummary import summary
# from Models.ViT import ViT
# from Models.ViT_luci import ViT_luci
# from pytorch_pretrained_vit import ViT

from source.datasets.ODIRDatasets import get_ODIRDataset
from params import *

if __name__ == '__main__':
    model_name = "Vit_Pretrained_ImageNet21K"
    save_dir = f'SavedModels/{model_name}/'

    train_ratio = 0.85
    batch_size = 8
    epochs = 2
    lr = 1e-2
    normalize_mean_resnet = [0.485, 0.456, 0.406]
    normalize_std_resnet = [0.229, 0.224, 0.225]
    normalize_mean_ViTPretrained = [0.5, 0.5, 0.5]
    normalize_std_ViTPretrained = [0.5, 0.5, 0.5]

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(save_dir): os.mkdir(save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=512).to(device)  # pretrained ViT
    model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=8).to(device)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation

    for name, p in model.named_parameters():
        # print(f'Layer {name}, grad: {p.requires_grad}')
        p.requires_grad = False
    for p in model.fc_norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    train_data_loader, val_data_loader, train_size, val_size, input_shape, num_classes = get_ODIRDataset(data_root, train_ratio, batch_size, normalize='transform',transform=transform)

    print("Model Summary: ")
    summary(model, input_size=input_shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.01, weight_decay=0.)  # parameter used in https://github.com/Zoe0123/Vision-Transformer-for-Chest-X-Ray-Classification/blob/main/vit.ipynb

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()  # from here: https://timm.fast.ai/
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.01, weight_decay=0.)  # parameter used in https://github.com/Zoe0123/Vision-Transformer-for-Chest-X-Ray-Classification/blob/main/vit.ipynb
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    training_history = train(model, optimizer, criterion, train_data_loader, val_data_loader, epochs, model_name, save_dir)

    plt.plot(training_history['loss_train'])
    plt.plot(training_history['loss_val'])
    plt.show()