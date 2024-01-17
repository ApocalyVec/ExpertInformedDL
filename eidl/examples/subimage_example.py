import pickle

import numpy as np

from eidl.Models.ViT_pretrained.pytorch_pretrained_vit.utils import get_sim_source_attention
from eidl.utils.model_utils import get_subimage_model


if __name__ == '__main__':

    # load image data ###########################################################
    # model and the image data will be downloaded when first used
    # find the best model in result directory
    subimage_handler = get_subimage_model(n_jobs=16)

    # you can either provide or not provide the source (human) attention as an argument to subimage_handler.compute_perceptual_attention(),
    # if not provided, the model attention will be returned otherwise the perceptual attention will be returned

    # compute the static attention for the given image
    rtn1 = subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', is_plot_results=True, discard_ratio=0.1, model_name='vit')

    # you can get the perceptual attenton if you provide the source attention
    human_attention = get_sim_source_attention(subimage_handler.image_data_dict['RLS_036_OS_TC']['original_image'])  # create a dummy source attention, it must be of the same size as the original image
    rtn2 = subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', source_attention=human_attention, discard_ratio=0.1, normalize_by_subimage=True, model_name='vit')

    # you can also get gradcam from one of the three cnn models (inception, vgg, resnet)
    rtn3 = subimage_handler.compute_perceptual_attention('RLS_036_OS_TC', is_plot_results=True, discard_ratio=0.1, model_name='resnet')