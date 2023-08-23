
import torch
import numpy as np
import matplotlib.pyplot as plt

from source.utils.model_utils import get_trained_model, load_image

# replace the image path to yours
image_path = r'D:\Dropbox\Dropbox\ExpertViT\Datasets\OCTData\oct_v2\reports_cleaned\G_Suspects\RLS_074_OD_TC.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device)

image_normalized, image = load_image(image_path, image_size, image_mean, image_std)

# show the image

plt.imshow(image)

# get the prediction
y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device), collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])
decoded_label = compound_label_encoder.decode(predicted_label)

print(f'Predicted label: {decoded_label}')