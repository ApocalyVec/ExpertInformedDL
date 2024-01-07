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
from eidl.utils.training_utils import train_oct_model

# User parameters ##################################################################################

# Change the following to the file path on your system #########
data_root = 'D:/Dropbox/Dropbox/ExpertViT/Datasets/OCTData/oct_v2'
results_dir = 'results'

n_jobs = 16  # n jobs for loading data from hard drive

# generic training parameters ##################################
epochs = 100
random_seed = 42
batch_size = 8
folds = 3

# grid search hyper-parameters ##################################
################################################################
# depths = 1, 3
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
# model_names = 'vit_small_patch32_224_in21k',
model_names = 'base',

################################################################
image_size = 1024, 512

# end of user parameters #############################################################################
if __name__ == '__main__':

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
    results_dir = f"{results_dir}-{dt_string}"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        print(f"Results will be save to {results_dir}")
    else:
        print(f"Results exist in {results_dir}, overwritting the results")

    # TODO train and test should use the same std and mean to normalize, moreover train and test should splitted during run time
    print("Creating data set")
    folds, test_dataset, image_means, image_stds = get_oct_test_train_val_folds(data_root, image_size=image_size, n_folds=folds, n_jobs=n_jobs)
    pickle.dump((image_means, image_stds), open(os.path.join(results_dir, 'image_means_stds.p'), 'wb'))
    pickle.dump(test_dataset.compound_label_encoder, open(os.path.join(results_dir, 'compound_label_encoder.p'), 'wb'))
    # test_dataset = get_oct_dataset(test_image_path, test_image_main, image_size=image_size, n_jobs=n_jobs)

    train_dataset, valid_dataset = folds[0]  # TODO using only one fold for now

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # save the data loader # TODO use test and save folds in the future
    pickle.dump(folds, open(os.path.join(results_dir, 'folds.p'), 'wb'))
    pickle.dump(test_dataset, open(os.path.join(results_dir, 'test_dataset.p'), 'wb'))

    parameters = set()
    for depth, alpha, model_name, lr, aoi_loss_dist in itertools.product(depths, alphas, model_names, lrs, aoi_loss_distance_types):
        if model_name != 'base':
            this_lr = lr * non_pretrained_lr_scaling
            this_depth = None  # depth does not affect the pretrained model
        else:
            this_lr = lr
            this_depth = depth
        parameters.add((this_depth, alpha, model_name, this_lr, aoi_loss_dist))

    for i, parameter in enumerate(parameters):  # iterate over the grid search parameters
        depth, alpha, model_name, lr, aoi_loss_dist = parameter
        model, grid_size = get_model(model_name, image_size=image_size, depth=depth, device=device)
        model_config_string = f'model-{model_name}_alpha-{alpha}_dist-{aoi_loss_dist}_depth-{model.depth}_lr-{lr}'
        print(f"Grid search [{i}] of {len(parameters)}: {model_config_string}")

        train_dataset.create_aoi(grid_size)
        valid_dataset.create_aoi(grid_size)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()
        train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = train_oct_model(
            model, model_config_string, train_loader, valid_loader, results_dir=results_dir, optimizer=optimizer, num_epochs=epochs,
            alpha=alpha, dist=aoi_loss_dist, l2_weight=None)

    # viz_oct_results(results_dir, test_image_path, test_image_main, batch_size, image_size, n_jobs=n_jobs)