import os.path

import numpy as np
import timm
import torch
from matplotlib import pyplot as plt

from eidl.datasets.ODIRDatasets import get_ODIRDataset
from eidl.utils.training_utils import train

# from Models.ViT import ViT
# from Models.ViT_luci import ViT_luci
# from pytorch_pretrained_vit import ViT

# Change the following to the file path on your system #########
data_root = 'D:/Dropbox/Dropbox/ExpertViT/Datasets/ODIR'
data_export_root = 'C:/Data/ODIR'
results_dir = 'archived/results_odir'

n_jobs = 16  # n jobs for loading data from hard drive

# generic training parameters ##################################
epochs = 500
random_seed = 42
batch_size = 8
folds = 3

# grid search hyper-parameters ##################################
################################################################
depths = 1, 3
# depths = 3,

################################################################
# alphas = 0.0, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.0
alphas = 1e-2, 0.0
# alphas = .0,

################################################################
# lrs = 1e-2, 1e-3, 1e-4
lrs = 1e-3, 1e-4
non_pretrained_lr_scaling = 1e-2

################################################################
aoi_loss_distance_types = 'Wasserstein', 'cross-entropy'

################################################################
# model_names = 'base', 'vit_small_patch32_224_in21k', 'vit_small_patch16_224_in21k', 'vit_large_patch16_224_in21k'
# model_names = 'base', 'vit_small_patch32_224_in21k'
# model_names = 'vit_base_patch32_384',
model_names = 'vit_large_patch32_384',
pretrained = True
# model_names = 'base',

train_size = 0.8

model_name = 'ViTNotPretrained'
save_dir = f'results-08_15_2023_15_43_28/SavedModels/{model_names[0]}-Pretrained_{pretrained}/'

if __name__ == '__main__':
    save_dir = f'{results_dir}/{model_name}/'

    # normalize_mean_resnet = [0.485, 0.456, 0.406]
    # normalize_std_resnet = [0.229, 0.224, 0.225]
    # normalize_mean_ViTPretrained = [0.5, 0.5, 0.5]
    # normalize_std_ViTPretrained = [0.5, 0.5, 0.5]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(save_dir): os.mkdir(save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=512).to(device)  # pretrained ViT

    # config = resolve_data_config({}, model=model)
    # transform = create_transform(**config)

    train_data_loader, val_data_loader, train_size, val_size, input_shape, num_classes = get_ODIRDataset(data_root, train_size, batch_size, data_export_root=data_export_root)
    model = timm.create_model(model_names[0], pretrained=pretrained, num_classes=8, img_size=input_shape[-1]).to(device)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
    # model = ViT_LSTM(image_size=input_shape[1:], num_patches=16, num_classes=8, embed_dim=512, dim_head=512, depth=6, heads=8, mlp_dim=2048, weak_interaction=False).to(device)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation

    # for name, p in model.named_parameters():
    #     # print(f'Layer {name}, grad: {p.requires_grad}')
    #     p.requires_grad = False
    # for p in model.fc_norm.parameters():
    #     p.requires_grad = True
    # for p in model.head.parameters():
    #     p.requires_grad = True

    # print("Model Summary: ")
    # summary(model, input_size=input_shape)

    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=1e-4, weight_decay=0.)  # parameter used in https://github.com/Zoe0123/Vision-Transformer-for-Chest-X-Ray-Classification/blob/main/vit.ipynb
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[0])
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()  # from here: https://timm.fast.ai/
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.01, weight_decay=0.)  # parameter used in https://github.com/Zoe0123/Vision-Transformer-for-Chest-X-Ray-Classification/blob/main/vit.ipynb
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    training_history = train(model, optimizer, train_data_loader, val_data_loader, epochs, model_name, save_dir, n_classes=8)

    plt.plot(training_history['loss_train'])
    plt.plot(training_history['loss_val'])
    plt.show()