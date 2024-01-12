"""
This file implements ViT using the cropped images


"""

import itertools
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from eidl.datasets.OCTDataset import get_oct_test_train_val_folds
from eidl.utils.iter_utils import collate_fn
from eidl.utils.model_utils import get_model
from eidl.utils.training_utils import train_oct_model, get_class_weight

# User parameters ##################################################################################

# Change the following to the file path on your system #########
# data_root = 'D:/Dropbox/Dropbox/ExpertViT/Datasets/OCTData/oct_v2'
data_root = r'C:\Dropbox\ExpertViT\Datasets\OCTData\oct_v2'
# data_root = r'C:\Users\apoca_vpmhq3c\Dropbox\ExpertViT\Datasets\OCTData\oct_v2'

cropped_image_data_path = r'C:\Dropbox\ExpertViT\Datasets\OCTData\oct_v2\oct_reports_info_repaired.p'
# results_dir = 'results'
# use_saved_folds = 'results-01_07_2024_10_53_56'

results_dir = '../temp/results'
# use_saved_folds = None
# use_saved_folds = '../temp/results-base-vit-lr-scheduler'
# use_saved_folds = '../temp/results-repaired-pretrained-vit'
use_saved_folds = '../temp/results-repaired-inception'


n_jobs = 20  # n jobs for loading data from hard drive and z-norming the subimages

# generic training parameters ##################################
epochs = 1
random_seed = 42
batch_size = 2
folds = 3

test_size = 0.1
val_size = 0.14

l2_weight = 1e-5

# grid search hyper-parameters ##################################
################################################################
# depths = 1, 3
# depths = 12,
depths = 1,

################################################################
# alphas = 0.0, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.0
# alphas = 1e-2, 0.0
alphas = 1e-2,
# alphas = .0,

################################################################
# lrs = 1e-2, 1e-3, 1e-4
# lrs = 1e-4, 1e-5
# lrs = 1e-4,
lrs = 1e-4,

non_pretrained_lr_scaling = 1e-2

################################################################
# aoi_loss_distance_types = 'Wasserstein', 'cross-entropy'
aoi_loss_distance_types = 'cross-entropy',

################################################################
# model_names = 'base', 'vit_small_patch32_224_in21k', 'vit_small_patch16_224_in21k', 'vit_large_patch16_224_in21k'
# model_names = 'base', 'vit_small_patch32_224_in21k'
# model_names = 'vit_small_patch32_224_in21k_subimage',
# model_names = 'base_subimage',
model_names = 'inception_v4_subimage',

grid_search_params = {
    'vit_small_patch32_224_in21k_subimage': {
        'alphas': 1e-2,
        'lrs': 1e-3,
        'aoi_loss_distance_types': 'cross-entropy',
        'optimizer': optim.SGD,
    },

    'base_subimage': {
        'alphas': 1e-2,
        'lrs': 1e-4,
        'aoi_loss_distance_types': 'cross-entropy',
        'depths': 1,
        'optimizer': optim.Adam,},

    'inception_v4_subimage': {
        'lrs': 1e-3,
        'optimizer': optim.SGD,
    }
}

################################################################
image_size = 1024, 512
patch_size = 32, 32
gaussian_smear_sigma = 0.5

# end of user parameters #############################################################################
if __name__ == '__main__':

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_saved_folds:
        print(f"Using saved folds from {use_saved_folds}")
        folds = pickle.load(open(os.path.join(use_saved_folds, 'folds.p'), 'rb'))
        test_dataset = pickle.load(open(os.path.join(use_saved_folds, 'test_dataset.p'), 'rb'))
        image_stats = pickle.load(open(os.path.join(use_saved_folds, 'image_stats.p'), 'rb'))
        test_dataset.compound_label_encoder = pickle.load(open(os.path.join(use_saved_folds, 'compound_label_encoder.p'), 'rb'))
        results_dir = use_saved_folds
    else:
        print("Creating data set")
        folds, test_dataset, image_stats = get_oct_test_train_val_folds(data_root, image_size=image_size, n_folds=folds, n_jobs=n_jobs,
                                                                                    cropped_image_data_path=cropped_image_data_path,
                                                                                    patch_size=patch_size, gaussian_smear_sigma=gaussian_smear_sigma,
                                                                                    test_size=test_size, val_size=val_size)
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
        results_dir = f"{results_dir}-{dt_string}"
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
            print(f"Results will be save to {results_dir}")
        else:
            print(f"Results exist in {results_dir}, overwritting the results")
        pickle.dump(folds, open(os.path.join(results_dir, 'folds.p'), 'wb'))
        pickle.dump(test_dataset, open(os.path.join(results_dir, 'test_dataset.p'), 'wb'))
        pickle.dump(image_stats, open(os.path.join(results_dir, 'image_stats.p'), 'wb'))
        pickle.dump(test_dataset.compound_label_encoder, open(os.path.join(results_dir, 'compound_label_encoder.p'), 'wb'))

    train_trial_dataset, valid_dataset, train_unique_img_dataset = folds[0]  # TODO using only one fold for now
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # TODO use the test loader in the future

    # process the grid search parameters
    parameters = set()
    # for model, grid_search_params

    for depth, alpha, model_name, lr, aoi_loss_dist in itertools.product(depths, alphas, model_names, lrs, aoi_loss_distance_types):
        # if model_name == 'pretrained':
        #     this_lr = lr * non_pretrained_lr_scaling
        #     this_depth = None  # depth does not affect the pretrained model
        # else:
        this_lr = lr
        this_depth = depth
        if 'inception' in model_name:  # inception net doesn't have depth and alpha
            this_params = (model_name, None, None, None, this_lr)
        else:
            this_params = (model_name, this_depth, alpha, aoi_loss_dist, this_lr)
        parameters.add(this_params)

    for i, parameter in enumerate(parameters):  # iterate over the grid search parameters
        model_name, depth, alpha, aoi_loss_dist, lr = parameter
        model = get_model(model_name, image_size=image_stats['subimage_sizes'], depth=depth, device=device, patch_size=patch_size)
        model_config_string = f"model-{model_name}_alpha-{alpha}_dist-{aoi_loss_dist}_lr-{lr}" + (f'_depth-{model.depth}' if hasattr(model, 'depth') else '')
        print(f"Grid search [{i}] of {len(parameters)}: {model_config_string}")

        if 'inception' in model_name or alpha == 0.0:
            train_dataset = train_unique_img_dataset
        else:
            train_dataset = train_trial_dataset
        train_dataset.create_aoi(use_subimages=True)
        valid_dataset.create_aoi(use_subimages=True)

        class_weights = get_class_weight(train_dataset.labels_encoded, 2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 5, T_mult=1, eta_min=1e-6, last_epoch=-1)
        scheduler = None

        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = train_oct_model(
            model, model_config_string, train_loader, valid_loader, results_dir=results_dir, optimizer=optimizer, num_epochs=epochs,
            alpha=alpha, dist=aoi_loss_dist, l2_weight=l2_weight, class_weights=class_weights)

    # viz_oct_results(results_dir, test_image_path, test_image_main, batch_size, image_size, n_jobs=n_jobs)