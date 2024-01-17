import os
import pickle
import tempfile
import urllib
from typing import Callable

import numpy as np
import timm
import torch
import gdown

from eidl.Models.ExpertAttentionViT import ViT_LSTM
from eidl.Models.ExpertAttentionViTSubImages import ViT_LSTM_subimage
from eidl.Models.ExtensionTimmViT import ExtensionTimmViT
from eidl.Models.ExtensionTimmViTSubimage import ExtensionTimmViTSubimage
from eidl.Models.ExtensionModel import ExtensionModelSubimage
from eidl.utils.image_utils import load_oct_image
from eidl.utils.iter_utils import reverse_tuple, chunker
from eidl.params import project_version


def get_model(model_name, image_size, depth, device, *args, **kwargs):
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
    elif model_name == 'vit_small_patch32_224_in21k':  # assuming any other name is timm models
        model = timm.create_model(model_name, img_size=reverse_tuple(image_size), pretrained=True, num_classes=2)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
        model = ExtensionTimmViT(model).to(device)
    elif model_name == 'vit_small_patch32_224_in21k_subimage':
        model = timm.create_model(model_name.replace('_subimage', ''),  pretrained=True, num_classes=2, dynamic_img_size=True)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
        model = ExtensionTimmViTSubimage(model).to(device)
    elif model_name == 'inception_v4_subimage':
        model = timm.create_model(model_name.replace('_subimage', ''),  pretrained=True, features_only=True)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
        model = ExtensionModelSubimage(model, num_classes=2, model_name=model_name).to(device)
    elif model_name == 'resnet50_subimage':
        model = timm.create_model('resnetv2_50x1_bit.goog_in21k', pretrained=True, num_classes=2, features_only=True)
        model = ExtensionModelSubimage(model, num_classes=2, model_name=model_name).to(device)
    elif model_name == 'vgg19_subimage':
        model = timm.create_model('vgg19.tv_in1k', pretrained=True, num_classes=2, features_only=True)
        model = ExtensionModelSubimage(model, num_classes=2, model_name=model_name).to(device)

    else:
        raise ValueError(f"model name {model_name} is not supported")
    return model


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
        temp = parameter_string.strip(f'{parameter_name}-')
        return 0. if temp == 'None' else float(temp)
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
    model_url = f"{github_file_url}/trained_model/best_model-base_alpha-0.01_dist-cross-entropy_depth-1_lr-0.0001_statedict_{model_param}.pt"
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
    model, grid_size = get_model(model_name, image_size=image_size, depth=depth, device=device)
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

def parse_epoch_metric_line(line, note=''):
    metrics = [np.nan if x == '' else float(x) for x in line.strip("training: ").strip("validation: ").split(",")]
    rtn = {}
    if len(metrics) == 2:
        rtn[f'{note}loss'], rtn[f'{note}acc'] = metrics
    elif len(metrics) == 6:
        rtn[f'{note}loss'], rtn[f'{note}acc'], rtn[f'{note}auc'], rtn[f'{note}precision'], rtn[f'{note}recall'], rtn[f'{note}f1'] = metrics
    else:
        raise ValueError(f"len(metrics) = {len(metrics)} is not supported")
    return rtn

def parse_training_results(results_dir):
    results_dict = {}
    config_strings = [i.strip('log_').strip('.txt') for i in os.listdir(results_dir) if i.startswith('log')]
    # columns = ['model_name', 'train acc', 'train loss', 'validation acc', 'validation loss', 'test acc']
    # results_df = pd.DataFrame(columns=columns)

    for i, c_string in enumerate(config_strings):
        print(f"Processing [{i}] of {len(config_strings)} configurations: {c_string}")
        model = torch.load(os.path.join(results_dir, f'best_{c_string}.pt'))  # TODO should load the model with smallest loss??
        with open(os.path.join(results_dir, f'log_{c_string}.txt'), 'r') as file:
            lines = file.readlines()
        results = []
        for epoch_lines in chunker(lines, 3):  # iterate three lines at a time
            results.append({**parse_epoch_metric_line(epoch_lines[1],  'train_'), **parse_epoch_metric_line(epoch_lines[2], 'val_')})
        results = np.array(results)
        # best_val_acc_epoch_index = np.argmax(results[:, 2])
        # test_acc = test_without_fixation(model, test_loader, device)  # TODO restore the test_acc after adding test method to extention
        # add viz pca of patch embeddings, attention rollout (gif and overlay), and position embeddings,
        # values = [model_config_string, *results[best_val_acc_epoch_index], test_acc]
        # results_df = pd.concat([results_df, pd.DataFrame(dict(zip(columns, values)))], ignore_index=True) # TODO fix the concat

        test_acc = None
        results_dict[c_string] = {'config_string': c_string,
                                  'train_accs': [x['train_acc'] for x in results],
                                  'train_losses': [x['train_loss'] for x in results],

                                  'val_accs': [x['val_acc'] for x in results],
                                  'val_losses': [x['val_loss'] for x in results],

                                  'train_aucs': [x['train_auc'] for x in results] if 'train_auc' in results[0] else None,
                                  'train_precisions': [x['train_precision'] for x in results] if 'train_precision' in results[0] else None,
                                  'train_recalls': [x['train_recall'] for x in results] if 'train_recall' in results[0] else None,
                                  'train_f1s': [x['train_f1'] for x in results] if 'train_f1' in results[0] else None,

                                  'val_aucs': [x['val_auc'] for x in results] if 'val_auc' in results[0] else None,
                                  'val_precisions': [x['val_precision'] for x in results] if 'val_precision' in results[0] else None,
                                  'val_recalls': [x['val_recall'] for x in results] if 'val_recall' in results[0] else None,
                                  'val_f1s': [x['val_f1'] for x in results] if 'val_f1' in results[0] else None,

                                  'test_acc': test_acc,
                                  'model': model}  # also save the model
    # results_df.to_csv(os.path.join(results_dir, "summary.csv"))
    return results_dict, config_strings


from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_subimage_model(*args, **kwargs):
    # make a temp dir with version number
    temp_dir = os.path.join(tempfile.gettempdir(), f"eidl_{project_version}")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # get the dataset
    print("Downloading the dataset...")
    data_file_id = '1ZvBpMvq92DxSGIzn-Cs0Vq37cYj77lWZ'
    data = download_and_load(data_file_id, temp_dir, _p_load)
    print("dataset downloaded and loaded.")

    # get the resnet model
    print("Downloading resnet model...")
    resnet_model = download_and_load('1oIvAaZM1SoKke4AUE_RBS7O85cIy_isP', temp_dir, torch.load)  # this is to load the model into the cache

    # get the vgg model
    print("Downloading vgg model...")
    vgg_model = download_and_load('1qWsKqfS_ym8HGioz4bMPvWfldGPxZHpI', temp_dir, torch.load)  # this is to load the model into the cache

    # get the vit model
    print("Downloading vit model...")
    vit_model = download_and_load('1SSMi74PwnIbGmzSz8X53-N58fYxKB2hU', temp_dir, torch.load)  # this is to load the model into the cache
    print("vit model downloaded and loaded.")
    patch_size = vit_model.patch_height, vit_model.patch_width

    # get the inception model
    print("Downloading inception model...")
    inception_model = download_and_load('1miWqj_UyS8QQYyRQqGBMzMiB02fRnhm0', temp_dir, torch.load)
    print("inception model downloaded and loaded.")


    # download the compound label encoder
    print("Downloading the compound label encoder...")
    compound_label_encoder = download_and_load('1K5xFlovm8hVX6EQLNZGw6Gn8CuLVnEwT', temp_dir, _p_load)
    print("compound label encoder downloaded and loaded.")

    # create the subimage handler
    from eidl.utils.SubimageHandler import SubimageHandler
    subimage_handler = SubimageHandler()
    subimage_handler.load_image_data(data, patch_size=patch_size, *args, **kwargs)
    subimage_handler.models['vit'] = vit_model
    subimage_handler.models['inception'] = inception_model
    subimage_handler.models['resnet'] = resnet_model
    subimage_handler.models['vgg'] = vgg_model
    subimage_handler.compound_label_encoder = compound_label_encoder

    return subimage_handler


def download_and_load(file_id: str, temp_dir: str, load_func: Callable):
    save_path = os.path.join(temp_dir, f"{file_id}")
    try:
        _gdown(file_id, save_path)
        model = load_func(save_path)
        return model

    except FileNotFoundError:
        print("Downloading the dataset failed because google drive has limited the download quota. \n"
              "Please download from this link https://drive.google.com/uc?id=1ZvBpMvq92DxSGIzn-Cs0Vq37cYj77lWZ,  \n"
              "Run this command to move the downloaded file to the temp directory:  \n"
              f"mv <path>/<to>/oct_reports_info_repaired.p {save_path}\n"
              "Then run your code again.")
        raise FileNotFoundError
    except Exception as e:
        print(f"Unable to download or load file with id {file_id} failed with error {e}")
        raise e

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} onto {device}")
    model = torch.load(model_path, map_location=device)
    return model

def _gdown(file_id, destination):
    """Download a Google Drive file identified by the file_id.
    Args:
        file_id (str): the file identifier.
        destination (str): the destination path.
    """
    if not os.path.exists(destination):
        gdown.download(id=file_id, output=destination, quiet=False)

def _p_load(path):
    return pickle.load(open(path, 'rb'))