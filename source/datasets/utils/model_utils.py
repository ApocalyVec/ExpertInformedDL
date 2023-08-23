import timm

from source.Models.ExpertAttentionViT import ViT_LSTM
from source.Models.ExtensionModels import ExpertTimmVisionTransformer


def get_vit_model(model_name, image_size, depth, device):
    if model_name == 'base':
        model = ViT_LSTM(image_size=reverse_tuple(image_size), num_patches=32, num_classes=2, embed_dim=128, depth=depth, heads=1,
                         mlp_dim=2048, weak_interaction=False).to(device)
    else:  # assuming any other name is timm models
        model = timm.create_model(model_name, img_size=reverse_tuple(image_size), pretrained=True, num_classes=2)  # weights from 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz', official Google JAX implementation
        model = ExpertTimmVisionTransformer(model).to(device)
    return model, model.get_grid_size()

def reverse_tuple(t):
    if len(t) == 0:
        return t
    else:
        return(t[-1],)+reverse_tuple(t[:-1])

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