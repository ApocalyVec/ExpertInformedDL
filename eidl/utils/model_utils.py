import os
import pickle
import tempfile
import urllib
from typing import Iterable

import timm
import torch

from eidl.Models.ExpertAttentionViT import ViT_LSTM
from eidl.Models.ExpertAttentionViTSubImages import ViT_LSTM_subimage
from eidl.Models.ExtensionModels import ExpertTimmVisionTransformer
from eidl.utils.image_utils import load_oct_image


def get_vit_model(model_name, image_size, depth, device, *args, **kwargs):
    # if type(image_size[0]) == int:
    #     image_size = swap_tuple(image_size, 0, -1)
    # if isinstance(image_size[0], Iterable):
    #     image_size = [swap_tuple(x, 0, -1) for x in image_size]
    if model_name == 'base':
        # model = ViT_LSTM(image_size=reverse_tuple(image_size), patch_size=(32, 16), num_classes=2, embed_dim=128, depth=depth, heads=1,
        #                  mlp_dim=2048, weak_interaction=False).to(device)
        model = ViT_LSTM(image_size=image_size, num_patches=32, num_classes=2, embed_dim=128, depth=depth, heads=1,
                         mlp_dim=2048, weak_interaction=False).to(device)
    elif model_name == 'base_subimage':
        model = ViT_LSTM_subimage(image_size=image_size, num_classes=2, embed_dim=128, depth=depth, heads=1,
                         mlp_dim=2048, weak_interaction=False, *args, **kwargs).to(device)  # NOTE, only this option supporst variable patch size
    elif model_name == 'pretrained':  # assuming any other name is timm models
        model = timm.create_model(model_name, img_size=reverse_tuple(image_size), pretrained=True, num_classes=2)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
        model = ExpertTimmVisionTransformer(model).to(device)
    else:
        raise ValueError(f"model name {model_name} is not supported")
    return model, model.get_grid_size()

def reverse_tuple(t):
    if len(t) == 0:
        return t
    else:
        return(t[-1],)+reverse_tuple(t[:-1])

def swap_tuple(t, i, j):
    t = list(t)
    t[i], t[j] = t[j], t[i]
    return tuple(t)

def parse_model_parameter(model_config_string: str, parameter_name: str):
    assert parameter_name in model_config_string
    parameter_string = [x for x in model_config_string.split('_') if parameter_name in x][0]
    parameter_value = parameter_string.split('-')[1]
    if parameter_name == 'dist':
        return parameter_string.strip(f'{parameter_name}-')
    elif parameter_name in ['alpha', 'dist', 'depth', 'lr']:
        return float(parameter_string.strip(f'{parameter_name}-'))
    elif parameter_name == 'model':
        return model_config_string[:model_config_string.find('_alpha')].split('-')[1]
    else:
        return parameter_value


def get_trained_model(device, model_param):
    """
    to use the model returned by this function, user should use model_utils.load_image and pass the returns (image_mean, image_std, image_size)
    as arguments.
    Parameters
    ----------
    device

    Returns
    a tuple of four items
    model: the trained model
    model_param: str: can be 'num-patch-32_image-size-1024-512', or 'patch-size-50-25_image-size-1000-500'
    image_mean: means of the RGB channels of the data on which the model is trained
    image_std: stds of the
    image_size: the size of the image used by the model
    -------

    """
    model_name = 'base'
    depth = 1

    if model_param == 'num-patch-32_image-size-1024-512':
        image_size = 1024, 512
    elif model_param == 'patch-size-50-25_image-size-1000-500':
        image_size = 1000, 500
    else:
        raise ValueError(f"model_param {model_param} is not supported")

    github_file_url = "https://raw.githubusercontent.com/ApocalyVec/ExpertInformedDL/master/trained_model/0.0.1"
    model_url = f"{github_file_url}/best_model-base_alpha-0.01_dist-cross-entropy_depth-1_lr-0.0001_statedict_{model_param}.pt"
    image_mstd_url = f"{github_file_url}/image_means_stds_{model_param}.p"
    compound_label_encoder_url = f"{github_file_url}/compound_label_encoder.p"

    temp_dir = tempfile.mkdtemp()
    model_file_path = os.path.join(temp_dir, "model_weights.pt")
    image_mstd_file_path = os.path.join(temp_dir, f"image_means_stds_{model_param}.p")
    compound_label_encoder_file_path = os.path.join(temp_dir, "compound_label_encoder.p")

    # Download the file using urlretrieve
    urllib.request.urlretrieve(model_url, model_file_path)
    urllib.request.urlretrieve(image_mstd_url, image_mstd_file_path)
    urllib.request.urlretrieve(compound_label_encoder_url, compound_label_encoder_file_path)

    print(f"File downloaded successfully and saved to {model_file_path}")
    model, grid_size = get_vit_model(model_name, image_size=image_size, depth=depth, device=device)
    model.load_state_dict(torch.load(model_file_path))

    image_mean, image_std = pickle.load(open(image_mstd_file_path, 'rb'))

    compound_label_encoder = pickle.load(open(compound_label_encoder_file_path, 'rb'))
    return model, image_mean, image_std, image_size, compound_label_encoder

def load_image_preprocess(image_path, image_size, image_mean, image_std):
    image = load_oct_image(image_path, image_size)
    image_normalized = (image - image_mean) / image_std
    # transpose to channel first
    image_normalized = image_normalized.transpose((2, 0, 1))
    return image_normalized, image