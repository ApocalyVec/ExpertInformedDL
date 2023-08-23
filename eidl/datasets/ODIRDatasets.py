import copy
import os.path
import pickle
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_image(image_path):
    return Image.open(image_path)

class ODIRDataset:
    """
    Ocular disease intelligent recognition dataset from: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
    Publication:
    A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-disease Detection
    https://link.springer.com/chapter/10.1007/978-3-030-71058-3_11
    """
    def __init__(self, dataset_root, image_size=None, n_jobs=16):
        """
        :param dataset_root: please set the dataset_root to the root of the extracted folder
        """
        # load the labels
        self.df = pd.read_csv(os.path.join(dataset_root, "full_df.csv"))

        # reload the entire dataset into memory for faster processing
        image_names = [(os.path.join(dataset_root, "preprocessed_images", row["filename"]), ) for index, row in self.df.iterrows()]
        self.labels = np.array([re.sub(r'\W+', '', row["labels"]) for index, row in self.df.iterrows()])
        print(f"Loading {len(image_names)} images with {n_jobs} jobs")
        start_time = time.time()
        with Pool(n_jobs) as p:
            self.img_data = p.starmap(load_image, image_names)
        print("Done loading images, took {} seconds".format(time.time() - start_time))
        # for index, row in self.df.iterrows():
        #     print("Loading {} of {} images".format(index + 1, len(self.df.index)), end='\r')
        #     img = Image.open(os.path.join(dataset_root, "preprocessed_images", row["filename"]))
        #     # img = np.moveaxis(img, -1, 0)
        #     label = row["labels"]
        #     self.img_data.append(img)
        #     self.labels.append(re.sub(r'\W+', '', label))
        # one-hot encode the labels
        self.num_class = len(np.unique(self.labels))
        # assert len(set([(x.shape[1]) for x in self.img_data] + [(x.shape[0]) for x in self.img_data])) == 1  # check image size
        self.image_size = image_size
        # self.img_data = np.array(self.img_data, dtype='uint8')  # set datatype to float32 to match that of torch's default datatype
        self.img_data_norm = None

        self.label_encoder = preprocessing.OneHotEncoder()
        self.onehot_encoded_labels = self.label_encoder.fit_transform(np.array(self.labels).reshape(-1, 1)).toarray()
        # print("Convert data to GPU tensor")
        # self._onehot_encoded_labels = torch.Tensor(self.onehot_encoded_labels).to(device)
        # self._img_data = torch.Tensor(self.img_data).to(device)

    def apply_subset(self, indices):
        rtn = copy.deepcopy(self)
        rtn.img_data = rtn.img_data[indices]
        rtn.img_data_norm = rtn.img_data_norm[indices]
        rtn.labels = rtn.labels[indices]
        rtn.onehot_encoded_labels = rtn.onehot_encoded_labels[indices]
        rtn.df = rtn.df.iloc[indices]
        return rtn

    def apply_transform(self, transform):
        self.img_data_norm = []
        print("Applying transform to images")
        for i in range(len(self.img_data)):
            self.img_data_norm.append(transform(self.img_data[i]))
            self.img_data[i] = np.array(self.img_data[i])
        self.img_data_norm = torch.stack(self.img_data_norm)

    def preprocess_all(self, normalize_mean=None, normalize_std=None, image_size=None):
        print("Normalizing input images")
        for i in range(len(self.img_data)):
            self.img_data[i] = np.array(self.img_data[i])
        self.img_data = np.array(self.img_data, dtype=np.float64)  # must NOT use float32, otherwise mean will be wronge due to overflow
        if normalize_std is None and normalize_mean is None:
            normalize_mean = np.mean(self.img_data.reshape(-1, self.img_data.shape[-1]), axis=0)
            normalize_std = np.std(self.img_data.reshape(-1, self.img_data.shape[-1]), axis=0)
        if image_size is not None:
            transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)])
        self.img_data_norm = []
        for i in range(len(self.img_data)):
            self.img_data_norm.append(transform(self.img_data[i]))
        self.img_data_norm = torch.stack(self.img_data_norm).to(torch.float32)

        # self.img_data = self.img_data / 255.  # normalize the uint8 pixels to between 0 and 1
        # if normalize_std is None and normalize_mean is None:
        #     means = np.mean(self.img_data, axis=(0, 2, 3), keepdims=True)
        #     stds = np.std(self.img_data, axis=(0, 2, 3), keepdims=True)
        # else:
        #     means = np.reshape(normalize_mean, newshape=[1, 3, 1, 1])
        #     stds = np.reshape(normalize_std, newshape=[1, 3, 1, 1])
        # self.img_data_norm = ((self.img_data - means) / stds).astype(np.float32)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        return {'image': self.img_data_norm[index], 'y': self.onehot_encoded_labels[index], 'image_original': self.img_data[index]}

def get_ODIRDataset(data_root, train_ratio, batch_size, data_export_root = None, normalize='data', normalize_mean=None, normalize_std=None, transform = None, image_size=None, random_seed=None):

    if data_export_root is not None and os.path.exists(dataset_path := os.path.join(data_export_root, 'ODIR_dataset.p')):
        print(f"Loading dataset from {dataset_path}")
        dataset = pickle.load(open(dataset_path, 'rb'))
    else:
        print(f"Creating dataset from {data_root}")
        dataset = ODIRDataset(data_root)
        if normalize == "data":
            dataset.preprocess_all()
        elif normalize == "provided":
            assert normalize_mean is not None and normalize_std is not None
            dataset.preprocess_all(normalize_mean=normalize_mean, normalize_std=normalize_std, image_size=image_size)
        elif normalize == "transform":
            assert transform is not None
            dataset.apply_transform(transform)
        else:
            raise Exception(f"Unrecognized normalize option {normalize}")
        if data_export_root is not None:
            dataset_path = os.path.join(data_export_root, 'ODIR_dataset.p')
            pickle.dump(dataset, open(dataset_path, 'wb'))

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    skf = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=random_seed)
    train_val_image_indices, test_image_indices = [(train, test) for train, test in skf.split(dataset.img_data, dataset.labels)][0]  # split by image labels, not trials!

    train_set = dataset.apply_subset(train_val_image_indices)
    val_set = dataset.apply_subset(test_image_indices)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_data_loader, val_data_loader, train_size, val_size, dataset[0]['image'].shape, dataset.num_class


def get_test_dataset(data_root, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    input_shape = image_datasets['train'][0][0].shape

    return dataloaders['train'], dataloaders['val'], dataset_sizes['train'], dataset_sizes['val'], input_shape, len(class_names)

if __name__ == "__main__":
    data_root = "D:/Dropbox/Dropbox/ExpertViT/Datasets/ODIR"
    dataset = ODIRDataset(data_root, device='cpu')