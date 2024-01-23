import os
import shutil
import tempfile
import time

import numpy as np

from eidl.utils.model_utils import get_subimage_model, count_parameters


def test_get_subimage_model_download():
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
    temp_dir = os.path.join(temp_dir, f"eidl")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    subimage_handler = get_subimage_model(n_jobs=1)

    count_parameters(subimage_handler.models['vit'])
    count_parameters(subimage_handler.models['inception'])
    count_parameters(subimage_handler.models['vgg'])
    count_parameters(subimage_handler.models['resnet'])

def test_get_subimage_model_multijob():
    n_jobs = 24
    start_time = time.time()
    subimage_handler_multijob = get_subimage_model(n_jobs=n_jobs)
    print(f"Time to load subimage handler with {n_jobs} jobs: {time.time() - start_time}")
    start_time = time.time()
    subimage_handler_singlejob = get_subimage_model(n_jobs=1)
    print(f"Time to load subimage handler with 1 job: {time.time() - start_time}")

    # check if the subimages are the same
    print("check multi vs single job resulted in the same subimages")
    for image_name, image_data in subimage_handler_multijob.image_data_dict.items():
        assert image_name in subimage_handler_singlejob.image_data_dict
        for i, s_image_data in enumerate(image_data['sub_images']):
            s_image_name = s_image_data['name']
            assert s_image_name == subimage_handler_singlejob.image_data_dict[image_name]['sub_images'][i]['name']
            assert np.all(s_image_data['image'] == subimage_handler_singlejob.image_data_dict[image_name]['sub_images'][i]['image'])

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
    subimage_handler = get_subimage_model(n_jobs=24)
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


def test_precompute_resnet():
    model_name = 'resnet'
    subimage_handler = get_subimage_model(precompute=model_name, n_jobs=24)
    for image_name in subimage_handler.image_data_dict.keys():
        assert (model_name, image_name) in subimage_handler.attention_cache


def test_precompute_vit():
    model_name = 'vit'
    discard_ratio = 0.1
    subimage_handler = get_subimage_model(precompute=model_name, n_jobs=24, discard_ratio=discard_ratio)
    for image_name in subimage_handler.image_data_dict.keys():
        assert (model_name, image_name, discard_ratio) in subimage_handler.attention_cache

def test_precompute_multiple_model():
    model_names = ['vit', 'resnet']
    discard_ratio = 0.1
    subimage_handler = get_subimage_model(precompute=model_names, n_jobs=24)
    for image_name in subimage_handler.image_data_dict.keys():
        assert ('resnet', image_name) in subimage_handler.attention_cache
        assert ('vit', image_name, discard_ratio) in subimage_handler.attention_cache
