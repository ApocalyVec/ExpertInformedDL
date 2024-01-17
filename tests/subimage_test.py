import os
import shutil
import tempfile

import numpy as np

from eidl.utils.model_utils import get_subimage_model, count_parameters
from eidl.params import project_version


def test_get_subimage_model():
    # delete the download files from the temp folder
    temp_dir = tempfile.gettempdir()

    # for 0.0.11 and older ####
    vit_path = os.path.join(temp_dir, "vit.pt")
    inception_path = os.path.join(temp_dir, "inception.pt")
    compound_label_encoder_path = os.path.join(temp_dir, "compound_label_encoder.p")
    dataset_path = os.path.join(temp_dir, "oct_reports_info.p")

    if os.path.exists(vit_path):
        os.remove(vit_path)
    if os.path.exists(inception_path):
        os.remove(inception_path)
    if os.path.exists(compound_label_encoder_path):
        os.remove(compound_label_encoder_path)
    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    #####
    temp_dir = os.path.join(temp_dir, f"eidl_{project_version}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    subimage_handler = get_subimage_model(n_jobs=16)

    count_parameters(subimage_handler.models['vit'])
    count_parameters(subimage_handler.models['inception'])
    count_parameters(subimage_handler.models['vgg'])
    count_parameters(subimage_handler.models['resnet'])


def test_vit_attention():
    subimage_handler = get_subimage_model(n_jobs=16)
    model_type = 'vit'
    image_name = 'RLS_036_OS_TC'
    discard_ratio = 0.1

    human_attention = np.zeros(subimage_handler.image_data_dict['RLS_036_OS_TC']['original_image'].shape[:2])
    human_attention[1600:1720, 2850:2965] = 1
    # compute the static attention for the given image
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=discard_ratio, normalize_by_subimage=True, model_name='vit')
    assert (model_type, image_name, discard_ratio) in subimage_handler.attention_cache
    subimage_handler.compute_perceptual_attention(image_name, source_attention=human_attention, discard_ratio=discard_ratio, normalize_by_subimage=True, model_name=model_type)



def test_gradcam():
    subimage_handler = get_subimage_model(n_jobs=16)
    image_name = 'RLS_036_OS_TC'
    # compute the static attention for the given image
    model_name = 'inception'
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)
    assert (model_name, image_name) in subimage_handler.attention_cache
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)

    model_name = 'vgg'
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)
    assert (model_name, image_name) in subimage_handler.attention_cache
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)

    model_name = 'resnet'
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)
    assert (model_name, image_name) in subimage_handler.attention_cache
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_name)

