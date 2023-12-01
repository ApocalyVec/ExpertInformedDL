import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import PIL
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from PIL import Image
from PIL import Image as im

from eidl.utils.image_utils import generate_image_binary_mask


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


def get_heatmap(seq, grid_size):
    """
    get the heatmap from the fixations
    Parameters
    ----------
    seq

    Returns
    -------

    """
    heatmap = np.zeros(grid_size)
    grid_width, grid_height = grid_size
    for i in seq:
        heatmap[int(np.floor(i[0] * grid_width)), int(np.floor(i[1] * grid_height))] += 1
    assert (heatmap.sum() == len(seq))
    heatmap = heatmap / heatmap.sum()
    assert abs(heatmap.sum() - 1) < 0.01, ValueError("no fixations sequence")
    return heatmap

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


def resize_image(image_name, image_size, image):
    image = load_oct_image(image, image_size)
    return image_name, {'image': image}

class OCTDatasetV2(Dataset):

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

        # process unique images
        if is_unique_images:  # keep on images with unique names in the trial samples
            unique_name_trial_samples = []
            unique_names = []
            for s in trial_samples:
                if s['name'] not in unique_names:
                    unique_name_trial_samples.append(s)
                    unique_names.append(s['name'])
            self.trial_samples = unique_name_trial_samples

    def create_aoi(self, grid_size):
        """
        aoi size is equal to (num_patches_width, num_patches_height). So it depends on the model
        Parameters
        ----------
        grid_size

        Returns
        -------

        """
        for i in range(len(self.trial_samples)):
            fixation_sequence = self.trial_samples[i]['fix_seq']
            if fixation_sequence is None:
                raise ValueError(f"image at index {i} does not have corresponding fixation seq.")
            aoi_from_fixation = get_heatmap(fixation_sequence, grid_size)
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


def collate_fn(batch):
    img = torch.stack([torch.FloatTensor(item['image_z_normed']) for item in batch], dim=0)
    # label = torch.LongTensor([item['label'] for item in batch])
    label = torch.IntTensor([item['label_encoded'] for item in batch])
    label_encoded = torch.FloatTensor([item['label_onehot_encoded'] for item in batch])
    # if np.any(np.array([item['seq'] for item in batch]) == None):
    #     fixation_sequence = None
    #     aoi_heatmap = None
    # else:
    fixation_sequence = [torch.FloatTensor(item['fix_seq']) for item in batch]
    aoi_heatmap = torch.stack([torch.FloatTensor(item['aoi']) for item in batch], dim=0)

    original_image = torch.stack([torch.FloatTensor(item['image']) for item in batch], dim=0)

    return img, label, label_encoded, fixation_sequence, aoi_heatmap, original_image


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

    Structure of the cropped_image_data:
        label: str:
            image_name: str:
                'image': np.array
                'sub_images':
                    'En-face_52.0micrometer_Slab_(Retina_View)':
                        'sub_image': np.array
                        'position': list of four two-int tuples
                    'Circumpapillary_RNFL':         same as above
                    'RNFL_Thickness_(Retina_View)': same as above
                    'GCL_Thickness_(Retina_View)':  same as above
                    'RNFL_Probability_and_VF_Test_points(Field_View)': same as above
                    'GCL+_Probability_and_VF_Test_points': same as above
                'label': str: 'G', 'S', 'G_Suspects', 'S_Suspects'

    Parameters
    ----------
    data_root
    image_size
    n_jobs

    Returns
    -------
    """
    # check if cropped image data exists
    if cropped_image_data_path is not None:
        cropped_image_data = pickle.load(open(cropped_image_data_path, 'rb'))
        # turn into a dict keyed by image name
        # cropped_image_data = {k: v for k, v in cropped_image_data.items() for k, v in v.items()}
    else:
        cropped_image_data = None

    image_root = os.path.join(data_root, 'reports_cleaned')
    assert os.path.exists(image_root), f"image directory {image_root} does not exist, please download the data from drive"

    pvalovia_dir = os.path.join(data_root, 'pvalovia-data')
    assert os.path.exists(pvalovia_dir), f"pvalovia directory {pvalovia_dir} does not exist, please download the data from github"

    image_dirs = os.listdir(image_root)
    name_label_images_dict = {}
    # get the images and labels from the image directories

    if cropped_image_data_path is None:
        for i, image_dir in enumerate(image_dirs):
            print(f"working on image directory {image_dir}, {i+1}/{len(image_dirs)}")
            label = image_dir[0]  # get the image label
            image_fns = os.listdir((this_image_dir := os.path.join(image_root, image_dir)))
            image_names = [n.split('.')[0] for n in image_fns]  # remove file extension
            load_image_args = [(os.path.join(this_image_dir, fn), image_size) for fn in image_fns]
            with Pool(n_jobs) as p:
                images = p.starmap(load_oct_image, load_image_args)
            name_label_images_dict = {**name_label_images_dict,
                                      **{image_name: {'name': image_name, 'image': image, 'label': label}
                                         for image_name, image in zip(image_names, images)}}
    else:
        load_image_args = [(image_name, image_size, image_info_dict['image']) for image_name, image_info_dict in cropped_image_data.items()]
        # change the key original_image to image
        with Pool(n_jobs) as p:
            name_label_images_dict = dict(p.starmap(resize_image, load_image_args))
        # change the key name of the image data from the original cropped_image_data from image to original image
        for k in cropped_image_data.keys():
            cropped_image_data[k]['original_image'] = cropped_image_data[k].pop('image')
        for k in cropped_image_data.keys():
            name_label_images_dict[k] = {**name_label_images_dict[k], **cropped_image_data[k]}

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
            trial_samples.append({**{'fix_seq': fixation_sequence}, **name_label_images_dict[image_name]})
            image_name_counts[image_name] += 1

    print(f"Number of trials without fixation sequence {no_fixation_count} with {len(trial_samples)} valid trials")
    plt.hist(image_name_counts.values())
    plt.xlabel("Number of trials")
    plt.ylabel("Number of images")
    plt.title("Number of trials per image")
    plt.show()
    print(f"Each image is used in on average:median {np.mean(list(image_name_counts.values()))}:{np.median(list(image_name_counts.values()))} trials")
    # plot the distribution of among trials and among images
    image_labels = np.array([v['label'] for v in name_label_images_dict.values()])
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
    plt.title("Number of images per label")
    plt.show()

    return trial_samples, name_label_images_dict, image_labels

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
    trial_samples, name_label_images_dict, image_labels = get_oct_data(data_root, image_size, n_jobs, *args, **kwargs)
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

    # compute white mask for each image
    for i in range(len(trial_samples)):
        trial_samples[i]['white_mask'] = generate_image_binary_mask(trial_samples[i]['image'], channel_first=False)

# TODO process the sub images
    # z-normalize the images
    image_data = np.array([x['image'] for x in trial_samples])
    image_means = np.mean(image_data, axis=(0, 1, 2))
    image_stds = np.std(image_data, axis=(0, 1, 2))
    for i in range(len(trial_samples)):
        trial_samples[i]['image_z_normed'] = (trial_samples[i]['image'] - image_means) / image_stds

    # make the image channel_first to be compatible with downstream training
    for i in range(len(trial_samples)):
        trial_samples[i]['image'] = trial_samples[i]['image'].transpose((2, 0, 1))
        trial_samples[i]['image_z_normed'] = trial_samples[i]['image_z_normed'].transpose((2, 0, 1))

    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_image_indices, test_image_indices = [(train, test) for train, test in skf.split(image_names, image_labels)][0]  # split by image labels, not trials!
    test_image_names = image_names[test_image_indices]

    test_trials = [x for x in trial_samples if x['name'] in test_image_names]  # get the trials matching the test images
    # check the label distro is stratified
    print(f"Test images has {[(unique_l, np.sum(np.array([img_l for img_l in image_labels[test_image_indices]]) == unique_l)) for unique_l in unique_labels]} labels")

    test_dataset = OCTDatasetV2(test_trials, is_unique_images=True, compound_label_encoder=compound_label_encoder)

    # now split the train and val with the remaining images

    train_val_image_names = image_names[train_val_image_indices]
    train_val_image_labels = image_labels[train_val_image_indices]
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

        folds.append([OCTDatasetV2(train_trials, is_unique_images=False, compound_label_encoder=compound_label_encoder),
                      OCTDatasetV2(val_trials, is_unique_images=True, compound_label_encoder=compound_label_encoder)])
    return folds, test_dataset, image_means, image_stds


