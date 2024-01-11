import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from eidl.utils.image_utils import generate_image_binary_mask, resize_image, load_oct_image, get_heatmap
from eidl.utils.SubimageHandler import SubimageHandler


def get_label(df_dir):
    if 'Healthy' in df_dir:
        label = 0
    elif 'Glaucoma' in df_dir:
        label = 1
    else:
        raise
    return label


def get_sequence(df_dir):
    df = pd.read_csv(df_dir)
    sequences = np.array(df[['norm_pos_x', 'norm_pos_y']])
    sequences[:, 1] = 1 - sequences[:, 1]
    return sequences


class OCTDatasetV3(Dataset):

    def __init__(self, trial_samples, is_unique_images, compound_label_encoder):
        """

        Parameters
        ----------
        trial_samples
        is_unique_images: bool: if true
        image_size
        """
        super().__init__()
        assert len(trial_samples) > 0

        self.compound_label_encoder = compound_label_encoder

        self.image_size = trial_samples[0]['image'].shape[1:]  # channel, width, height
        self.trial_samples = trial_samples
        self.labels_encoded = torch.tensor([x['label_encoded'] for x in trial_samples])
        # process unique images
        if is_unique_images:  # keep on images with unique names in the trial samples
            unique_name_trial_samples = []
            unique_names = []
            for s in trial_samples:
                if s['name'] not in unique_names:
                    unique_name_trial_samples.append(s)
                    unique_names.append(s['name'])
            self.trial_samples = unique_name_trial_samples

    def has_subimages(self):
        return 'sub_images' in self.trial_samples[0].keys()

    def create_aoi(self, grid_size=None, use_subimages=False):
        """
        aoi size is equal to (num_patches_width, num_patches_height). So it depends on the model
        Parameters
        ----------

        image_size: tuple of int, width and height of the image, must be provided when use_subimages is True

        Returns
        -------

        """
        for i in range(len(self.trial_samples)):
            fixation_sequence = self.trial_samples[i]['fix_seq']
            if self.has_subimages():
                image_size = self.trial_samples[i]['original_image'].shape[:-1]
                aoi_from_fixation = []
                for s_image in self.trial_samples[i]['sub_images']:

                    # keep the fixation sequence only to the within the subimage position
                    grid = s_image['mask'].shape  # the grid size for an image is the same as the mask size
                    percentage_position = [(x/image_size[1], y/image_size[0]) for x, y in s_image['position']]

                    if len(fixation_sequence) > 0:
                        s_image_fix_sequence = np.array([(x, y) for x, y in fixation_sequence if percentage_position[0][0] <= x <= percentage_position[2][0]
                                                                         and percentage_position[0][1] <= y <= percentage_position[2][1]])

                        # plt.imshow(self.trial_samples[i]['original_image'])
                        # fix_positions = np.array([(x * image_size[1], y * image_size[0]) for x, y in fixation_sequence])
                        # plt.scatter(fix_positions[:, 0], fix_positions[:, 1], c='r', s=1)
                        # fix_positions = np.array([(x * image_size[1], y * image_size[0]) for x, y in s_image_fix_sequence])
                        # plt.scatter(fix_positions[:, 0], fix_positions[:, 1], c='b', s=4)
                        # plt.show()

                        # normalize the subimage fixation sequence with respect to its grid size

                        if len(s_image_fix_sequence) > 0:
                            s_image_fix_seq_normed = np.zeros_like(s_image_fix_sequence)
                            s_image_fix_seq_normed[:, 0] = (s_image_fix_sequence[:, 0] - percentage_position[0][0]) / (percentage_position[2][0] - percentage_position[0][0])
                            s_image_fix_seq_normed[:, 1] = (s_image_fix_sequence[:, 1] - percentage_position[0][1]) / (percentage_position[2][1] - percentage_position[0][1])
                            aoi_heatmap_subimage = get_heatmap(s_image_fix_seq_normed, grid_size=grid, normalize=False)
                        else:
                            aoi_heatmap_subimage = np.zeros(grid)
                    else:
                        aoi_heatmap_subimage = np.ones(grid)
                    # plt.imshow(aoi_heatmap)
                    # plt.show()
                    aoi_from_fixation.append(aoi_heatmap_subimage.reshape(-1))
                aoi_from_fixation = np.concatenate(aoi_from_fixation)
                if aoi_from_fixation.sum() > 0:
                    aoi_from_fixation = aoi_from_fixation / aoi_from_fixation.sum()  # normalize globally
                if np.isnan(aoi_from_fixation).any():
                    raise ValueError(f"aoi heatmap contains nan at index {i}")
            else:
                if fixation_sequence is None:
                    raise ValueError(f"Not implemented for without subimages: image at index {i} does not have corresponding fixation seq.")
                aoi_from_fixation = get_heatmap(fixation_sequence, grid_size=grid_size)
            self.trial_samples[i]['aoi'] = aoi_from_fixation

    def __len__(self):
        return len(self.trial_samples)

    def __getitem__(self, index):
        return self.trial_samples[index]
        # image = Image.open(self.imgs[index]).convert('RGB')
        # image = image.crop((0, 0, 5360, 2656))
        # image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        # image /= 255
        # for d in range(3):
        #     image[d] = (image[d] - self.means[d]) / self.stds[d]
        # return {'img': image,
        #         'label': self.labels[index],
        #         'seq': self.sequences[index],
        #         'heatmap': self.heatmaps[index]}


def minmax_norm(x):
    x = x.copy()
    x[:, 0] = (x[:, 0] - min(x[:, 0])) / (max(x[:, 0]) - min(x[:, 0]))
    x[:, 1] = (x[:, 1] - min(x[:, 1])) / (max(x[:, 1]) - min(x[:, 1]))
    x[x == 1] -= 10 ** -6
    return x

def de_z_norm(x, mean, std):
    x = x.copy()
    assert x.shape[0] == 3
    for d in range(3):
        x[d] = x[d] * std[d] + mean[d]
    return x

def get_oct_data(data_root, image_size, n_jobs=1, cropped_image_data_path=None, *args, **kwargs):
    """
    expects two folds in data root:
        reports_cleaned: folds must have the first letter being either S or G (oct_labels)
        pvalovia-data

    Structure of the cropped_image_data: dict
            image_name: str: image names are the keys of the dict
                'image': np.array: the original image
                'sub_images': dict
                    'En-face_52.0micrometer_Slab_(Retina_View)':
                        'sub_image': np.array
                        'position': list of four two-int tuples
                    'Circumpapillary_RNFL':                             same as above
                    'RNFL_Thickness_(Retina_View)':                     same as above
                    'GCL_Thickness_(Retina_View)':                      same as above
                    'RNFL_Probability_and_VF_Test_points(Field_View)':  same as above
                    'GCL+_Probability_and_VF_Test_points':              same as above
                'label': str: 'G', 'S', 'G_Suspects', 'S_Suspects'

    Parameters
    ----------
    data_root
    image_size
    n_jobs

    Returns
    -------
    """
    pvalovia_dir = os.path.join(data_root, 'pvalovia-data')

    # check if cropped image data exists
    subimage_loader = SubimageHandler()
    if cropped_image_data_path is None:
        # get the images and labels from the image directories
        image_root = os.path.join(data_root, 'reports_cleaned')
        assert os.path.exists(
            image_root), f"image directory {image_root} does not exist, please download the data from drive"

        assert os.path.exists(
            pvalovia_dir), f"pvalovia directory {pvalovia_dir} does not exist, please download the data from github"
        image_dirs = os.listdir(image_root)
        image_data_dict = {}
        for i, image_dir in enumerate(image_dirs):
            print(f"working on image directory {image_dir}, {i+1}/{len(image_dirs)}")
            label = image_dir[0]  # get the image label
            image_fns = os.listdir((this_image_dir := os.path.join(image_root, image_dir)))
            image_names = [n.split('.')[0] for n in image_fns]  # remove file extension
            load_image_args = [(os.path.join(this_image_dir, fn), image_size) for fn in image_fns]
            with Pool(n_jobs) as p:
                images = p.starmap(load_oct_image, load_image_args)
            image_data_dict = {**image_data_dict,
                                      **{image_name: {'name': image_name, 'image': image, 'label': label}
                                         for image_name, image in zip(image_names, images)}}
    else:
        subimage_data = pickle.load(open(cropped_image_data_path, 'rb'))
        image_data = subimage_loader.load_image_data(subimage_data, n_jobs=n_jobs, *args, **kwargs)

        load_image_args = [(image_name, image_size, image_info_dict['original_image']) for image_name, image_info_dict in subimage_data.items()]
        with Pool(n_jobs) as p:
            image_data_dict = dict(p.starmap(resize_image, load_image_args))  # this dict contains the resized image

        for k in image_data.keys():
            image_data_dict[k] = {**image_data_dict[k], **image_data[k]}  # merge the two dicts


    # perform z-norm
    # compute white mask for each image
    for k, x in image_data_dict.items():
        image_data_dict[k]['white_mask'] = generate_image_binary_mask(x['image'], channel_first=False)

    # z-normalize the images
    # the z-normal should be computed from the 177 images, not from the 455 trials
    image_data = np.array([x['image'] for k, x in image_data_dict.items()])
    image_means = np.mean(image_data, axis=(0, 1, 2))
    image_stds = np.std(image_data, axis=(0, 1, 2))
    for k, x in image_data_dict.items():
        image_data_dict[k]['image_z_normed'] = (x['image'] - image_means) / image_stds

    # make the image channel_first to be compatible with downstream training
    for k, x in image_data_dict.items():
        image_data_dict[k]['image_z_normed'] = image_data_dict[k]['image_z_normed'].transpose((2, 0, 1))

    trial_samples = []
    image_name_counts = defaultdict(int)
    # load gaze sequences
    fixation_dirs = os.listdir(pvalovia_dir)
    no_fixation_count = 0
    for i, fixation_dir in enumerate(fixation_dirs):
        print(f"working on fixation directory {fixation_dir}, {i+1}/{len(fixation_dirs)}")
        this_fixation_dir = os.path.join(pvalovia_dir, fixation_dir)
        fixation_fns = [fn for fn in os.listdir(this_fixation_dir) if fn.endswith('.csv')]
        for fixation_fn in fixation_fns:
            image_name = fixation_fn.split('.')[0]
            image_name = image_name[image_name.find("_", image_name.find("_") + 1)+1:]
            fixation_sequence = get_sequence(os.path.join(this_fixation_dir, fixation_fn))
            # trials without fixation sequence are not included
            if len(fixation_sequence) == 0:
                no_fixation_count += 1
                continue
            trial_samples.append({**{'name': image_name, 'fix_seq': fixation_sequence}, **image_data_dict[image_name]})
            image_name_counts[image_name] += 1

    # add the images that doesn't have a trial
    trial_samples_image_names = [x['name'] for x in trial_samples]
    for image_name, image_data in image_data_dict.items():
        if image_name not in trial_samples_image_names:
            trial_samples.append({**{'name': image_name, 'fix_seq': np.zeros((0, 2))}, **image_data})

    print(f"Number of trials without fixation sequence {no_fixation_count} with {len(trial_samples)} valid trials")
    plt.hist(image_name_counts.values(), bins=np.arange(max(image_name_counts.values())))
    plt.xlabel("Number of trials")
    plt.ylabel("Number of images")
    plt.title("Number of trials per image")
    plt.show()
    print(f"Each image is used in on average:median {np.mean(list(image_name_counts.values()))}:{np.median(list(image_name_counts.values()))} trials")
    # plot the distribution of among trials and among images
    image_labels = np.array([v['label'] for v in image_data_dict.values()])
    unique_labels = np.unique(image_labels)

    plt.bar(np.arange(len(unique_labels)), [np.sum(image_labels==l) for l in unique_labels])
    plt.xlabel("Number of images")
    plt.xticks(np.arange(len(unique_labels)), unique_labels)
    plt.title("Number of images per label")
    plt.show()

    trial_labels = np.array([v['label'] for v in trial_samples])
    plt.bar(np.arange(len(unique_labels)), [np.sum(trial_labels==l) for l in unique_labels])
    plt.xlabel("Number of images")
    plt.xticks(np.arange(len(unique_labels)), unique_labels)
    plt.title("Number of trials per label")
    plt.show()

    # change suspect label to certain label
    for trial in trial_samples:
        if 'G_Suspects' in trial['label']:
            trial['label'] = 'G'
        elif 'S_Suspects' in trial['label']:
            trial['label'] = 'S'

    for image_name, image_data in image_data_dict.items():
        if 'G_Suspects' in image_data['label']:
            image_data['label'] = 'G'
        elif 'S_Suspects' in image_data['label']:
            image_data['label'] = 'S'
    image_labels = np.array([v['label'] for v in image_data_dict.values()])

    return trial_samples, image_data_dict, image_labels, {'image_means': image_means, 'image_stds': image_stds,
                                                          'subimage_mean': subimage_loader.subimage_mean, 'subimage_std': subimage_loader.subimage_std,
                                                          'subimage_sizes': [x['image'].shape[1:] for x in trial_samples[0]['sub_images']]}

class CompoundLabelEncoder:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.one_hot_encoder = preprocessing.OneHotEncoder()


    def fit_transform(self, labels):
        encoded_labels = self.label_encoder.fit_transform(labels)
        one_hot_encoded_labels = self.one_hot_encoder.fit_transform(encoded_labels.reshape(-1, 1)).toarray()
        return encoded_labels, one_hot_encoded_labels

    def encode(self, labels, one_hot=True):
        encoded_labels = self.label_encoder.transform(labels)
        if one_hot:
            one_hot_encoded_labels = self.one_hot_encoder.transform(encoded_labels.reshape(-1, 1)).toarray()
            return encoded_labels, one_hot_encoded_labels
        else:
            return encoded_labels

    def decode(self, encoded_labels):
        # check if the label is one hot encoded
        if len(encoded_labels.shape) == 2:
            assert encoded_labels.shape[1] == len(self.label_encoder.classes_), f"encoded labels shape {encoded_labels.shape} does not match the number of classes {len(self.label_encoder.classes_)}"
            encoded_labels = np.argmax(encoded_labels, axis=1)
        return self.label_encoder.inverse_transform(encoded_labels)



def get_oct_test_train_val_folds(data_root, image_size, n_folds, test_size=0.1, val_size=0.1, n_jobs=1, random_seed=None, *args, **kwargs):
    """
    we have two set of samples: image and trials

    trials can have duplicate images
    for test and val, we need to split by the number of images instead of trials

    Parameters
    ----------
    data_root
    image_size
    test_size
    val_size
    n_jobs
    random_seed

    Returns
    -------

    """
    trial_samples, name_label_images_dict, image_labels, image_stats = get_oct_data(data_root, image_size, n_jobs, *args, **kwargs)
    unique_labels = np.unique(image_labels)

    # create label and one-hot encoder
    compound_label_encoder = CompoundLabelEncoder()
    labels = np.array([x['label'] for x in trial_samples])
    encoded_labels, one_hot_encoded_labels = compound_label_encoder.fit_transform(labels)

    # add encoded labels to the trial samples
    for i, encoded_l, onehot_l in zip(range(len(trial_samples)), encoded_labels, one_hot_encoded_labels):
        trial_samples[i]['label_encoded'] = encoded_l
        trial_samples[i]['label_onehot_encoded'] = onehot_l

    image_names = np.array(list(name_label_images_dict.keys()))

    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_image_indices, test_image_indices = [(train, test) for train, test in skf.split(image_names, image_labels)][0]  # split by image labels, not trials!
    test_image_names = image_names[test_image_indices]

    test_trials = [x for x in trial_samples if x['name'] in test_image_names]  # get the trials matching the test images
    # check the label distro is stratified
    print(f"Test images has {[(unique_l, np.sum(np.array([img_l for img_l in image_labels[test_image_indices]]) == unique_l)) for unique_l in unique_labels]} labels")

    test_dataset = OCTDatasetV3(test_trials, is_unique_images=True, compound_label_encoder=compound_label_encoder)

    # now split the train and val with the remaining images

    train_val_image_names = image_names[train_val_image_indices]
    train_val_image_labels = image_labels[train_val_image_indices]
    val_size = val_size / (1 - test_size)  # adjust the val size to be relative to the train_val set
    skf = StratifiedShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
    folds = []
    for f_index, (train_image_indices, val_image_indices) in enumerate(skf.split(train_val_image_names, train_val_image_labels)):
        train_image_names = train_val_image_names[train_image_indices]
        val_image_names = train_val_image_names[val_image_indices]
        train_trials = [x for x in trial_samples if x['name'] in train_image_names]  # get the trials matching the test images
        val_trials = [x for x in trial_samples if x['name'] in val_image_names]  # get the trials matching the test images
        print( f"Fold {f_index}, train images has {[(unique_l, np.sum(np.array([img_l for img_l in train_val_image_labels[train_image_indices]]) == unique_l)) for unique_l in unique_labels]} labels")
        print( f"                train TRIALS has {[(unique_l, np.sum(np.array([x['label'] for x in train_trials]) == unique_l)) for unique_l in unique_labels]} labels")

        print( f"Fold {f_index}, val images has {[(unique_l, np.sum(np.array([img_l for img_l in train_val_image_labels[val_image_indices]]) == unique_l)) for unique_l in unique_labels]} labels")

        folds.append([OCTDatasetV3(train_trials, is_unique_images=False, compound_label_encoder=compound_label_encoder),
                      OCTDatasetV3(val_trials, is_unique_images=True, compound_label_encoder=compound_label_encoder),
                      OCTDatasetV3(train_trials, is_unique_images=True, compound_label_encoder=compound_label_encoder)])
    return folds, test_dataset, image_stats


