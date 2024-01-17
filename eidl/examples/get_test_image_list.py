import os
import pickle
import tempfile

import numpy as np
import torch
import matplotlib.pyplot as plt

from eidl.Models.ViT_pretrained.pytorch_pretrained_vit.utils import get_sim_source_attention
from eidl.datasets.OCTDataset import OCTDatasetV3
from eidl.utils.SubimageHandler import SubimageHandler
from eidl.utils.model_utils import get_best_model, parse_training_results, parse_model_parameter

patch_size=(32, 32)

# results_dir = 'results-01_05_2024_10_56_13'
# results_dir = '../temp/results-repaired-base-vit'
# results_dir = '../temp/results-repaired-pretrained-vit'
# results_dir = '../temp/results-repaired-inception'
# results_dir = '../temp/results-repaired-vgg'
results_dir = '../temp/results-repaired-resnet'


# figure_notes = 'square depth 1'
# figure_notes = 'static aggregated-self discard 0.1 '
figure_notes = 'test '
if __name__ == '__main__':

    # load model ###############################################################
    # find the best model in result directory
    results_dict, model_config_strings = parse_training_results(results_dir)
    models = {parse_model_parameter(x, 'model') for x in model_config_strings}
    best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict, fold=0)
    # # save the torch model
    # os.makedirs('../temp/trained_model/0.0.15')
    # torch.save(best_model, '../temp/trained_model/0.0.15/resnet.pt')

    # load image data ###########################################################

    test_dataset = pickle.load(open(os.path.join(results_dir, 'test_dataset.p'), 'rb'))
    folds = pickle.load(open(os.path.join(results_dir, 'folds.p'), 'rb'))

    valid_dataset = folds[0][1]  # TODO using only one fold for now
    test_dataset = OCTDatasetV3([*test_dataset.trial_samples, *valid_dataset.trial_samples], True,valid_dataset.compound_label_encoder)

    test_dataset_labels = np.array([x['label'] for x in test_dataset.trial_samples])
    print(f"Test dataset size: {len(test_dataset)} after combining with validation set, with {np.sum(test_dataset_labels == 'G')} glaucoma and {np.sum(test_dataset_labels == 'S')} healthy samples")

    # print the image names
    g_names = [x['name'] for x in test_dataset.trial_samples if x['label'] == 'G']
    s_names = [x['name'] for x in test_dataset.trial_samples if x['label'] == 'S']
    print(f"G names [{len(g_names)}]: {g_names}")
    print(f"S names [{len(s_names)}]: {s_names}")