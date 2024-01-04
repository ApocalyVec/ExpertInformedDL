import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from eidl.utils.iter_utils import collate_fn, collate_subimages
from eidl.utils.image_utils import preprocess_subimages, z_norm_subimages, process_aoi
from eidl.utils.torch_utils import any_image_to_tensor
from eidl.viz.vit_rollout import VITAttentionRollout
from eidl.viz.viz_oct_results import register_cmap_with_alpha
from eidl.viz.viz_utils import plot_subimage_rolls, plot_image_attention


class SubimageHandler:

    def __init__(self):
        self.subimage_mean = None
        self.subimage_std = None
        self.image_data_dict = None
        self.model = None


    def load_image_data(self, image_data_dict, *args, **kwargs):
        """

        Parameters
        ----------
        image_data_dict: dict
            image_name: str: image names are the keys of the dict
                'image': np.array: the original image
                'sub_images': dict
                    'En-face_52.0micrometer_Slab_(Retina_View)': str
                        'sub_image': np.array
                        'position': list of four two-int tuples
                    'Circumpapillary_RNFL':                             same as above
                    'RNFL_Thickness_(Retina_View)':                     same as above
                    'GCL_Thickness_(Retina_View)':                      same as above
                    'RNFL_Probability_and_VF_Test_points(Field_View)':  same as above
                    'GCL+_Probability_and_VF_Test_points':              same as above
                'label': str: 'G', 'S', 'G_Suspects', 'S_Suspects'

        Returns
        -------
        dict:
            image_name: str: image names are the keys of the dict
                label: str
                original_image: ndarray
                subimages: list of dict
                    dict keys:
                        image: ndarray
                        mask: ndarray
                        position list of four size-two tuples
                        name: subimage name

        """

        # change the key name of the image data from the original cropped_image_data from image to original image
        for k in image_data_dict.keys():
            image_data_dict[k]['original_image'] = image_data_dict[k].pop('image')

        # preprocess the subimages
        image_data_dict = preprocess_subimages(image_data_dict, *args, **kwargs)

        # process the subimages if there are any
        print("z norming subimages")
        image_data_dict, self.subimage_mean, self.subimage_std = z_norm_subimages(image_data_dict)

        for k, x in image_data_dict.items():
            for s_image_name, s_image_data in image_data_dict[k]['sub_images'].items():
                image_data_dict[k]['sub_images'][s_image_name]['sub_image_cropped_padded_z_normed'] = s_image_data[
                    'sub_image_cropped_padded_z_normed'].transpose((2, 0, 1))

        # get rid of the extra fields
        subimage_names = list(image_data_dict[list(image_data_dict.keys())[0]]['sub_images'].keys())
        for image_name, image_data in image_data_dict.items():
            subimages = image_data.pop('sub_images')
            image_data['sub_images'] = []
            for s_image_name in subimage_names:
                image_data['sub_images'].append(
                    {'image': subimages[s_image_name]['sub_image_cropped_padded_z_normed'],
                     'mask': subimages[s_image_name]['patch_mask'],
                     'position': subimages[s_image_name]['position'],
                     'name': s_image_name})
        self.image_data_dict = image_data_dict
        return image_data_dict

    def compute_perceptual_attention(self, image_name, source_attention=None, overlay_alpha=0.75, is_plot_results=True, save_dir=None,
                                     *args, **kwargs):
        """

        Parameters
        ----------
        image_name: name of the image in the image data dict
        source_attention: default None, ndarray: the human attention with which the perceptual attention will be computed.
                        if not provided, the model attention will be returned
        is_plot_results: if True, the results will be plotted, see the parameter save_dir
        save_dir: if provided, the plots will be saved to this directory instead of being shown

        Returns
        -------

        """
        assert self.model is not None, "model must be provided by setting it to the model attribute of the SubimageHandler class"
        assert image_name in self.image_data_dict.keys(), f"image name {image_name} is not in the image data dict"
        sample = self.image_data_dict[image_name]
        if source_attention is not None:
            assert source_attention.shape == sample['original_image'].shape[:-1], f"source attention shape {source_attention.shape} does not match image shape {sample['original_image'].shape[:-1]}"
        image_original_size = sample['original_image'].shape[:-1]

        device = next(self.model.parameters()).device
        patch_size = self.model.patch_height, self.model.patch_width

        # run the model on the image
        image, *_ = collate_subimages([sample])
        image = any_image_to_tensor(image, device)
        subimage_masks = [x[0].detach().cpu().numpy() for x in image['masks']]  # the masks for the subimages in a a single image
        subimages = [x[0].detach().cpu().numpy() for x in image['subimages']]  # the subimages in a single image
        subimage_positions = [x['position'] for x in sample['sub_images']]

        vit_rollout = VITAttentionRollout(self.model, device=device, attention_layer_name='attn_drop', *args, **kwargs)
        attention = vit_rollout(depth=self.model.depth, in_data=image, fixation_sequence=None)

        # get the subimage attention from the source
        rollout_image, subimage_roll = process_aoi(attention, image_original_size, True,
                                                   grid_size=self.model.get_grid_size(),
                                                   subimage_masks=subimage_masks, subimages=subimages,
                                                   subimage_positions=subimage_positions, patch_size=patch_size)

        if source_attention is not None:
            for s_image in sample['sub_images']:
                s_source_attention = source_attention[s_image['position'][0][0]:s_image['position'][1][0],
                                     s_image['position'][0][1]:s_image['position'][1][1]]
                pass
        else:
            source_attention = rollout_image
        if is_plot_results is not None:
            image_original = sample['original_image']
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
            cmap_name = register_cmap_with_alpha('viridis')
            plot_image_attention(image_original, rollout_image, source_attention, cmap_name,
                                 notes=f'{image_name}', overlay_alpha=overlay_alpha, save_dir=save_dir)
            plot_subimage_rolls(subimage_roll, subimages, subimage_positions, self.subimage_std, self.subimage_mean,
                                cmap_name, notes=f"{image_name}", overlay_alpha=overlay_alpha, save_dir=save_dir)
        return {"original_image_attention": rollout_image, "subimage_attention": subimage_roll, "subimage_position": subimage_positions}
