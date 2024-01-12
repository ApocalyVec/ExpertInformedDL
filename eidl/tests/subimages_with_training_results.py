import os
import pickle
import tempfile

import torch

from eidl.utils.SubimageHandler import SubimageHandler
from eidl.utils.model_utils import get_best_model, parse_training_results, parse_model_parameter

patch_size=(32, 32)
# data_path = 'C:/Users/apoca_vpmhq3c/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info_repaired.p'
data_path = 'C:/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info_repaired.p'
# results_dir = 'results-01_05_2024_10_56_13'
# results_dir = '../temp/results-repaired-base-vit'
results_dir = '../temp/results-repaired-inception'
source_attention_path = r"../temp/source_attention/GCL Prob RLS_036_OS_TC.pickle"
figure_dir = '../temp/figures_example/RLS_036_OS_TC'
model_type = 'inception'

# figure_notes = 'square depth 1'
# figure_notes = 'static aggregated-self discard 0.1 '
figure_notes = 'test '
if __name__ == '__main__':
    # load sample human attention ###################
    human_attention = pickle.load(open(source_attention_path, 'rb'))

    # load image data ###########################################################
    # the image data must comply with the format specified in SubimageLoader

    # check if the subimage handler is in the temp directory
    if os.path.exists(os.path.join(tempfile.gettempdir(), 'subimage_handler.p')):
        subimage_handler = pickle.load(open(os.path.join(tempfile.gettempdir(), 'subimage_handler.p'), 'rb'))
    else:
        data = pickle.load(open(data_path, 'rb'))
        compound_label_encoder = pickle.load(open(os.path.join(results_dir, 'compound_label_encoder.p'), 'rb'))
        subimage_handler = SubimageHandler()
        subimage_handler.compound_label_encoder = compound_label_encoder
        subimage_handler.load_image_data(data, patch_size=patch_size)
        # save the subimage handler to temp directory
        pickle.dump(subimage_handler, open(os.path.join(tempfile.gettempdir(), 'subimage_handler.p'), 'wb'))

    # load model ###############################################################
    # find the best model in result directory
    results_dict, model_config_strings = parse_training_results(results_dir)
    models = {parse_model_parameter(x, 'model') for x in model_config_strings}
    best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)
    # # save the torch model
    # os.makedirs('../temp/trained_model/0.0.9')
    # torch.save(best_model, '../temp/trained_model/0.0.9/vit.pt')
    subimage_handler.models[model_type] = best_model

    subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', discard_ratio=0.1, notes=figure_notes, normalize_by_subimage=True, model_name=model_type)
    subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', source_attention=human_attention, discard_ratio=0.1, notes=figure_notes, normalize_by_subimage=True, model_name=model_type)
    # subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', save_dir=figure_dir, discard_ratio=0.1, notes=figure_notes)
    # subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', source_attention=human_attention, save_dir='figures_example', discard_ratio=0.7,notes=figure_notes)
    # subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', save_dir='figures_example', discard_ratio=0.1,notes=figure_notes)
