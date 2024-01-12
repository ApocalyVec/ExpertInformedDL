import os
import tempfile

import numpy as np
import pytest

from eidl.utils.model_utils import get_subimage_model, count_parameters


def test_get_subimage_model():
    # delete the download files from the temp folder
    temp_dir = tempfile.gettempdir()
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

    subimage_handler = get_subimage_model(n_jobs=16)

    count_parameters(subimage_handler.models['vit'])
    count_parameters(subimage_handler.models['inception'])


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
    model_type = 'inception'
    image_name = 'RLS_036_OS_TC'
    # compute the static attention for the given image
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_type)
    assert (model_type, image_name) in subimage_handler.attention_cache
    subimage_handler.compute_perceptual_attention(image_name, discard_ratio=0.1, normalize_by_subimage=True, model_name=model_type)



