import pickle

from eidl.utils.image_utils import SubimageLoader

patch_size=(32, 32)
data_path = '/Users/apocalyvec/Dropbox/ExpertViT/Datasets/OCTData/oct_v2/oct_reports_info.pickle'


if __name__ == '__main__':
    data = pickle.load(open(data_path, 'rb'))

    subimage_loader = SubimageLoader()

    SubimageLoader.load_image_data(data, patch_size=patch_size, n_jobs=16)