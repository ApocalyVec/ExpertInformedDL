# Expert-attention guided deep learning for medical images

## Get Started


### Use the trained model for inference

Pip install the PYPI distro:

```bash
pip install expert-informed-dl
```

Check out example.py for a simple example of how to use the trained model for inference.

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

