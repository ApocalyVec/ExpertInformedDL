import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt


def plt2arr(fig, draw=True):
    """
    need to draw if figure is not drawn yet
    """
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr

def plot_train_history(history, note=''):
    plt.plot(history['train_accs'])
    plt.plot(history['val_accs'])
    plt.title('model accuracy ' + note)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    plt.show()

    # summarize history for loss
    plt.plot(history['train_losses'])
    plt.plot(history['val_losses'])
    plt.title('model loss ' + note)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    plt.show()


def plot_subimage_rolls(subimage_roll, subimages, subimage_positions, image_std, image_mean, cmap_name,
                        notes='', overlay_alpha=0.75, save_dir=None):
    for s_i, (s_roll, s_image, s_pos) in enumerate(zip(subimage_roll, subimages, subimage_positions)):
        # unznorm the image
        s_image_unznormed = np.transpose(s_image, (1, 2, 0)) * image_std + image_mean
        s_image_unznormed = s_image_unznormed.astype(np.uint8)
        s_image_unznormed = cv2.cvtColor(s_image_unznormed, cv2.COLOR_BGR2RGB)
        s_image_size = s_pos[2][1] - s_pos[0][1], s_pos[2][0] - s_pos[0][0]
        s_image_unznormed = s_image_unznormed[:s_image_size[0], :s_image_size[1]]

        # plot the aoi and subimage side by side, using subplot
        s_fig = plt.figure(figsize=(15, 10), constrained_layout=True)

        plt.subplot(1, 3, 1)
        plt.imshow(s_image_unznormed, cmap=cmap_name)
        plt.imshow(s_roll, cmap=cmap_name, alpha=overlay_alpha * s_roll / np.max(s_roll))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(s_image_unznormed)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(s_roll, cmap=cmap_name)
        plt.colorbar()
        plt.axis('off')

        plt.suptitle(title_text := f'{notes} Subimage {s_i}')

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'{title_text}.png'))
        else:
            plt.show()

def plot_image_attention(image_original, model_attention, source_attention, cmap_name, overlay_alpha=0.9, save_dir=None, notes=''):
    fig = plt.figure(figsize=(30, 20), constrained_layout=True)

    plt.subplot(2, 2, 1)
    plt.imshow(image_original)  # plot the original image
    if source_attention is not None:
        plt.imshow(source_attention, cmap=cmap_name, alpha=overlay_alpha * source_attention / np.max(source_attention))
    plt.axis('off')
    plt.title("Source Attention Overlay")

    if source_attention is not None:
        plt.subplot(2, 2, 3)
        plt.imshow(source_attention, cmap=cmap_name)
        plt.axis('off')
        plt.title("Source Attention")

    plt.subplot(2, 2, 2)
    plt.imshow(image_original)  # plot the original image
    plt.imshow(model_attention, cmap=cmap_name, alpha=overlay_alpha * model_attention / np.max(model_attention))
    plt.axis('off')
    plt.title("Model Attention Overlay")

    plt.subplot(2, 2, 4)
    plt.imshow(model_attention, cmap=cmap_name)
    plt.axis('off')
    plt.title("Model Attention")

    plt.suptitle(notes)
    # plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'{notes}.png'))
    else:
        plt.show()
