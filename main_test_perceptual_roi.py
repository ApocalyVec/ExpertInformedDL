import os
import pickle
import tempfile

import torch
import matplotlib.pyplot as plt

from eidl.Models.ViT_pretrained.pytorch_pretrained_vit.utils import get_sim_source_attention
from eidl.utils.SubimageHandler import SubimageHandler
from eidl.utils.model_utils import get_best_model, parse_training_results, parse_model_parameter

patch_size=(32, 32)
# data_path = 'C:/Users/apoca_vpmhq3c/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info_repaired.p'
data_path = 'C:/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info_repaired.p'

results_dir = '../temp/results-repaired-base-vit'

source_attention_path = r"../temp/perceptual_roi\samples"

figure_output = '../temp/perceptual_roi/norm CLS numpy rollout'
model_type = 'vit'

image_name = 'RLS_036_OS_TC'


if __name__ == '__main__':

    # load model ###############################################################
    # find the best model in result directory
    results_dict, model_config_strings = parse_training_results(results_dir)
    models = {parse_model_parameter(x, 'model') for x in model_config_strings}
    best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)

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
    subimage_handler.models[model_type] = best_model

    for i, fn in enumerate(os.listdir(source_attention_path)):
        print(f"Processing {fn} {i} of {len(os.listdir(source_attention_path))}")
        perceptual_aoi_info = pickle.load(open(os.path.join(source_attention_path, fn), 'rb'))
        source_attention = perceptual_aoi_info['gaze_attention_map']
        image_name = perceptual_aoi_info['image_name']

        subimage_handler.compute_perceptual_attention(image_name, source_attention=source_attention,
                                                      discard_ratio=0.1, notes=f"Sample-{i}_", normalize_by_subimage=False, model_name=model_type,
                                                      save_dir=figure_output, save_separate_overlay=True)

