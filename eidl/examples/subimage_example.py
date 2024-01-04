import pickle
from eidl.utils.model_utils import  get_subimage_model


if __name__ == '__main__':

    # load image data ###########################################################
    # model and the image data will be downloaded when first used
    # find the best model in result directory
    subimage_handler = get_subimage_model()
    # load sample human attention ###################
    human_attention = pickle.load(open(r"C:\Users\apoca\Downloads\9025_OD_2021_widefield_report Sample 2 in test set, original image.pickle", 'rb'))

    subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', source_attention=human_attention, is_plot_results=True, discard_ratio=0.1)