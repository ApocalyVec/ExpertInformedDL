import pytest

from eidl.utils.model_utils import get_subimage_model


def test_get_subimage_model():
    # delete the download files from the temp folder
    subimage_handler = get_subimage_model(n_jobs=16)


def test_vit_attention():
    subimage_handler = get_subimage_model(n_jobs=16)
    # compute the static attention for the given image
    rtn = subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', is_plot_results=True,
                                                        discard_ratio=0.9, model_name='vit')
    assert rtn is not None


def test_vit_attention_with_source():
    pass


def test_gradcam():
    pass

def test_attention_retention_subimage_handler():
    pass


