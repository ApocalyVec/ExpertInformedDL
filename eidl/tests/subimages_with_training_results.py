import pickle

import torch

from eidl.utils.SubimageHandler import SubimageHandler
from eidl.utils.model_utils import get_best_model, parse_training_results, parse_model_parameter

patch_size=(32, 32)
data_path = 'C:/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info.p'
results_dir = 'results-01_03_2024_22_31_25'


if __name__ == '__main__':

    # load image data ###########################################################
    # the image data must comply with the format specified in SubimageLoader
    data = pickle.load(open(data_path, 'rb'))
    subimage_handler = SubimageHandler()
    subimage_handler.load_image_data(data, patch_size=patch_size)

    # load model ###############################################################
    # find the best model in result directory
    results_dict, model_config_strings = parse_training_results(results_dir)
    models = {parse_model_parameter(x, 'model') for x in model_config_strings}
    best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)
    # # save the torch model
    torch.save(best_model, 'trained_model/0.0.2/best_model.pt')

    # load sample human attention ###################
    human_attention = pickle.load(open(r"C:\Users\apoca\Downloads\9025_OD_2021_widefield_report Sample 2 in test set, original image.pickle", 'rb'))

    subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', best_model, source_attention=human_attention, save_dir='figures_example', discard_ratio=0.1)