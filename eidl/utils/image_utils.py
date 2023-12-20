import PIL
import cv2
import numpy as np
from PIL import Image, Image as im
import matplotlib.pyplot as plt

def generate_image_binary_mask(image, channel_first=False):
    if channel_first:
        image = np.moveaxis(image, 0, -1)
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 254, 1, cv2.THRESH_BINARY_INV)
    return binary_mask

def z_normalize_image(image, mean, std):
    assert image.shape[-1] == 3, "Image should be in channel last format"
    image = image.astype(np.float32)
    image -= mean
    image /= std
    return image


def resize_image(image_name, image_size, image):
    image = load_oct_image(image, image_size)
    return image_name, {'image': image}


def load_oct_image(image_info, image_size):
    """

    @param image_info:
        if str, interpret as the image path,
        if dict, interpret as the image info dict, comes with the cropped image data
    """
    if isinstance(image_info, str):
        image = Image.open(image_info).convert('RGB')
    elif isinstance(image_info, np.ndarray):
        image = image_info
    else:
        raise ValueError(f"image info {image_info} is not a valid type")
    # image = image.crop((0, 0, 5360, 2656))
    # image = image.crop((0, 0, 5120, 2640))
    image = im.fromarray(image).resize(image_size, resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32)
    return image

def crop_image(image, patch_size):
    """
    crop the image into patches of size patch_size
    @param image:
    @param patch_size:
    @return:
    """
    image_size = image.shape[:2]
    # crop the image into patches of size patch_size
    n_patch_rows = image_size[0] // patch_size[0]
    n_patch_cols = image_size[1] // patch_size[1]
    return image[:n_patch_rows * patch_size[0], :n_patch_cols * patch_size[1]]  # crop from the bottom right corner


def pad_image(image, max_n_patches, patch_size):

    image_size = image.shape[:2]

    # image shape must be divisible by patch size before padding
    assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, f"image shape {image_size} is not divisible by patch size {patch_size}"
    n_patches = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

    # pad the image to the max size
    pad_size = ((max_n_patches[0] - n_patches[0]) * patch_size[0], (max_n_patches[1] - n_patches[1]) * patch_size[1])
    image_padded = np.pad(image, ((0, pad_size[0]), (0, pad_size[1]), (0, 0)), mode='constant', constant_values=0)

    patch_mask = np.ones((max_n_patches[0], max_n_patches[1]), dtype=bool)
    patch_mask[n_patches[0]:, :] = False
    patch_mask[:, n_patches[1]:] = False

    return image_padded, patch_mask


def pad_subimages(cropped_image_data, patch_size=(32, 32)):
    """
    sub image pad to max size
        'En-face_52.0micrometer_Slab_(Retina_View)':
        'Circumpapillary_RNFL':
        'RNFL_Thickness_(Retina_View)':
        'GCL_Thickness_(Retina_View)':
        'RNFL_Probability_and_VF_Test_points(Field_View)':
        'GCL+_Probability_and_VF_Test_points':

    first crop the image to the closest size divisible by the patch size, pad each sub image to the same size, so that we can batchify them

    Parameters
    ----------
    cropped_image_data
    patch_size: tuple of int, width and height of the patch

    Returns
    -------

    """
    image_names = list(cropped_image_data.keys())
    sub_image_names = list(cropped_image_data[image_names[0]]['sub_images'].keys())

    counter = 0
    for i, s_image_name in enumerate(sub_image_names):
        sub_images = {image_name: (image_data['sub_images'][s_image_name]['sub_image'], image_data['sub_images'][s_image_name]['position']) for image_name, image_data in cropped_image_data.items()}
        max_size = max([s_image.shape[:2] for (s_image, _) in sub_images.values()])
        max_size = (max_size[0] // patch_size[0] * patch_size[0], max_size[1] // patch_size[1] * patch_size[1])
        max_n_patches = (max_size[0] // patch_size[0], max_size[1] // patch_size[1])

        print(f"resizing sub-images {s_image_name}, {i + 1}/{len(sub_image_names)}, they will be cropped&padded to {max_size}, with {max_n_patches} patches ({patch_size=})")
        # find the max patchifiable size, round down

        for image_name, (s_image, position) in sub_images.items():
            temp = crop_image(s_image, patch_size)

            cropped_image_data[image_name]['sub_images'][s_image_name]['sub_image_cropped_padded'], \
                cropped_image_data[image_name]['sub_images'][s_image_name]['patch_mask'] = pad_image(temp, max_n_patches, patch_size)
            cropped_image_data[image_name]['sub_images'][s_image_name]['position'] = position
            # plt.imsave(f'C:/Users/apoca/Downloads/temp/{counter}_{s_image_name}_Aoriginal_subimage.png', s_image)
            # plt.imsave(f'C:/Users/apoca/Downloads/temp/{counter}_{s_image_name}_Bimage_cropped.png', temp)
            # plt.imsave(f'C:/Users/apoca/Downloads/temp/{counter}_{s_image_name}_Cimage_padded.png', cropped_image_data[image_name]['sub_images'][s_image_name]['sub_image_cropped_padded'])
            # plt.imsave(f'C:/Users/apoca/Downloads/temp/{counter}_{s_image_name}_Dpatch_mask.png', cropped_image_data[image_name]['sub_images'][s_image_name]['patch_mask'])
            #
            # counter += 1
    return cropped_image_data


def z_norm_subimages(name_label_images_dict):
    image_names = list(name_label_images_dict.keys())
    sub_image_names = list(name_label_images_dict[image_names[0]]['sub_images'].keys())

    all_sub_images = [image_data['sub_images'][s_image_name]['sub_image'] for image_name, image_data in name_label_images_dict.items() for i, s_image_name in enumerate(sub_image_names)]

    mean_values = np.stack([np.mean(image, axis=(0, 1)) for image in all_sub_images])
    all_mean = np.mean(mean_values, axis=0)

    std_values = np.stack([np.std(image, axis=(0, 1)) for image in all_sub_images])
    all_std = np.sqrt(np.mean(np.square(std_values), axis=0))

    # now normalize the sub images
    for image_name, image_data in name_label_images_dict.items():
        for s_image_name, s_image_data in image_data['sub_images'].items():
            s_image_data['sub_image_cropped_padded_z_normed'] = z_normalize_image(s_image_data['sub_image_cropped_padded'], all_mean, all_std)
            # s_image_data['sub_image_cropped_padded_z_normed'] = (s_image_data['sub_image_cropped_padded'] - all_mean) / all_std

    return name_label_images_dict, all_mean, all_std


def get_heatmap(seq, grid_size, normalize=True):
    """
    get the heatmap from the fixations
    Parameters

    grid_size: tuple of ints
    patch_size:
    ----------
    seq

    Returns
    -------

    """
    heatmap = np.zeros(grid_size)
    grid_height, grid_width = grid_size
    for i in seq:
        heatmap[int(np.floor(i[1] * grid_height)), int(np.floor(i[0] * grid_width))] += 1
    assert (heatmap.sum() == len(seq))
    if normalize:
        heatmap = heatmap / heatmap.sum()
        assert abs(heatmap.sum() - 1) < 0.01, ValueError("no fixations sequence")
    return heatmap


def remap_subimage_attention_rolls(rolls, subimage_masks, subsubimage_positions, original_image_size):
    print("remapping subimage attention rolls")