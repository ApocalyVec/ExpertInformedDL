# Expert-attention guided deep learning for medical images

## Get Started

Pip install the PYPI distro:

```bash
pip install expert-informed-dl
```

### Here's an example of how to use the trained model for inference (with subimages)

Check out eidl/examples/subimage_example.py for a simple example of how to use the trained model for inference on subimages.

```python
from eidl.utils.model_utils import get_subimage_model

subimage_handler = get_subimage_model()
subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', is_plot_results=True, discard_ratio=0.1)

```


### If you don't want to use subimages:

Check out eidl/examples/example.py for a simple example of how to use the trained model for inference.

When forwarding image through the network, use the argument `collapse_attention_matrix=True` to get the attention matrix
to get the attention matrix averaged across all heads and keys for each query token. 

```python
y_pred, attention_matrix = model(image_data, collapse_attention_matrix=False)

```


### Train model locally
Install `requirements.txt`

Download Pytorch matching with a CUDA version matching your GPU from [here](https://pytorch.org/get-started/locally/). 

Run `train.py`


For example, if you have 32 * 32 patches,
the attention matrix will be of size (32 * 32 + 1) 1025. Plus one for the classificaiton token.
If you set `collapse_attention_matrix=False`, the attention matrix will be
uncollapsed. The resulting attention matrix will be of shape (n_batch, n_heads, n_queries, n_keys). For example, if you have 32 * 32 patches,
one image and one head, the attention matrix will be of shape (1, 1, 1025, 1025).


## Troubleshoot

If get model functions raises the following error:

```bash
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

You will need to install the correct version of Pytorch matching with a CUDA version matching your GPU from [here](https://pytorch.org/get-started/locally/).
This is because all the models are trained on GPU.