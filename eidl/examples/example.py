import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from eidl.utils.model_utils import get_trained_model, load_image_preprocess
from eidl.viz.vit_rollout import VITAttentionRollout

# replace the image path to yours
image_path = r'D:\Dropbox\Dropbox\ExpertViT\Datasets\OCTData\oct_v2\reports_cleaned\G_Suspects\RLS_074_OD_TC.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device, model_param='num-patch-32_image-size-1024-512')

image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)

# get the prediction
image_tensor = torch.Tensor(image_normalized).unsqueeze(0).to(device)
y_pred, attention_matrix = model(image_tensor, collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])
decoded_label = compound_label_encoder.decode(predicted_label)

print(f'Predicted label: {decoded_label}')

# plot the attention rollout
vit_rollout = VITAttentionRollout(model, device=device, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.5)
rollout = vit_rollout(depth=model.depth, input_tensor=image_tensor)  # rollout on the last layer

rollout_resized = cv2.resize(rollout, dsize=image_size, interpolation=cv2.INTER_LINEAR)
rollout_heatmap = cv2.applyColorMap(cv2.cvtColor((rollout_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)
rollout_heatmap = cv2.cvtColor(rollout_heatmap, cv2.COLOR_BGR2RGB)
alpha = 0.2
output_image = cv2.addWeighted(image.astype(np.uint8), alpha, rollout_heatmap, 1 - alpha, 0)


fig = plt.figure(figsize=(15, 30), constrained_layout=True)
axes = fig.subplots(3, 1)
axes[0].imshow(image.astype(np.uint8))  # plot the original image
axes[0].axis('off')
axes[0].set_title(f'Original image')

axes[1].imshow(rollout_heatmap)  # plot the attention rollout
axes[1].axis('off')
axes[1].set_title(f'Attention rollout')

axes[2].imshow(output_image)  # plot the attention rollout
axes[2].axis('off')
axes[2].set_title(f'Overlayed attention rollout')
plt.show()
