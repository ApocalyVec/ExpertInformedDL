import os
import pickle
import tempfile
import urllib

import numpy as np
import timm
import torch
import gdown

from eidl.Models.ExpertAttentionViT import ViT_LSTM
from eidl.Models.ExpertAttentionViTSubImages import ViT_LSTM_subimage
from eidl.Models.ExtensionModels import ExpertTimmVisionTransformer
from eidl.utils.image_utils import load_oct_image
from eidl.utils.iter_utils import reverse_tuple, chunker


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

def get_subimage_model():
    github_file_url = "https://raw.githubusercontent.com/ApocalyVec/ExpertInformedDL/master/trained_model/0.0.2"
    model_url = f"{github_file_url}/model.pt"
    temp_dir = tempfile.gettempdir()
    model_file_path = os.path.join(temp_dir, "subimage_model.pt")
    dataset_path = os.path.join(temp_dir, "oct_reports_info.p")

    if not os.path.exists(model_file_path):
        # Download the file using urlretrieve
        urllib.request.urlretrieve(model_url, model_file_path)
    model = torch.load(model_file_path)

    patch_size = model.patch_height, model.patch_width

    # get the dataset
    if not os.path.exists(dataset_path):
        print("Downloading the dataset")
        gdown.download("https://drive.google.com/uc?id=1a-UcpqLGV7xRjZ-JXRIZ0p65nzJOfkHf", output=dataset_path, quiet=False)
    from eidl.utils.SubimageHandler import SubimageHandler
    data = pickle.load(open(dataset_path, 'rb'))
    subimage_handler = SubimageHandler()
    subimage_handler.load_image_data(data, patch_size=patch_size)
    subimage_handler.model = model

    return subimage_handler


def load_image_preprocess(image_path, image_size, image_mean, image_std):
    image = load_oct_image(image_path, image_size)
    image_normalized = (image - image_mean) / image_std
    # transpose to channel first
    image_normalized = image_normalized.transpose((2, 0, 1))
    return image_normalized, image

def get_best_model(models, results_dict):
    models = list(reversed(list(models)))
    best_model, best_model_results, best_model_config_string = None, None, None
    for model in models:  # get the best model each model architecture
        # model = 'vit_small_patch32_224_in21k'
        # model = 'base'
        best_model_val_acc = -np.inf
        best_model_config_string = None
        best_model_results = None
        for model_config_string, results in results_dict.items():
            this_val_acc = np.max(results['val_accs'])
            if parse_model_parameter(model_config_string, 'model') == model and this_val_acc > best_model_val_acc:
                best_model_val_acc = this_val_acc
                best_model_config_string = model_config_string
                best_model_results = results

        print(f"Best model for {model} has val acc of {best_model_val_acc} with parameters: {best_model_config_string}")
        best_model = best_model_results['model']
    return best_model, best_model_results, best_model_config_string

def parse_training_results(results_dir):
    results_dict = {}
    model_config_strings = [i.strip('log_').strip('.txt') for i in os.listdir(results_dir) if i.startswith('log')]
    # columns = ['model_name', 'train acc', 'train loss', 'validation acc', 'validation loss', 'test acc']
    # results_df = pd.DataFrame(columns=columns)

    for i, model_config_string in enumerate(model_config_strings):
        print(f"Processing [{i}] of {len(model_config_strings)} model configurations: {model_config_string}")
        model = torch.load(
            os.path.join(results_dir,
                         f'best_{model_config_string}.pt'))  # TODO should load the model with smallest loss??
        with open(os.path.join(results_dir, f'log_{model_config_string}.txt'), 'r') as file:
            lines = file.readlines()

        results = []
        for epoch_lines in chunker(lines, 3):  # iterate three lines at a time
            train_loss, train_acc = [np.nan if x == '' else float(x) for x in epoch_lines[1].strip("training: ").split(",")]
            val_loss, val_acc = [np.nan if x == '' else float(x) for x in epoch_lines[2].strip('validation: ').split(",")]
            results.append((train_acc, train_loss, val_acc, val_loss))
        results = np.array(results)
        # best_val_acc_epoch_index = np.argmax(results[:, 2])
        # test_acc = test_without_fixation(model, test_loader, device)  # TODO restore the test_acc after adding test method to extention
        # add viz pca of patch embeddings, attention rollout (gif and overlay), and position embeddings,
        # values = [model_config_string, *results[best_val_acc_epoch_index], test_acc]
        # results_df = pd.concat([results_df, pd.DataFrame(dict(zip(columns, values)))], ignore_index=True) # TODO fix the concat
        test_acc = None
        results_dict[model_config_string] = {'model_config_string': model_config_string, 'train_accs': results[:, 0], 'train_losses': results[:, 1], 'val_accs': results[:, 2], 'val_losses': results[:, 3], 'test_acc': test_acc, 'model': model}  # also save the model
    # results_df.to_csv(os.path.join(results_dir, "summary.csv"))
    return results_dict, model_config_strings
