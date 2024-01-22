import numpy as np
import torch

def gaussian_filter(shape, center, sigma=1.0, normalized=True):
    """
    Create a Gaussian matrix with a given shape and center location.

    Parameters:
        shape (tuple): Shape of the matrix (rows, columns).
        center (tuple): Center location of the Gaussian (center_row, center_col).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: The Gaussian matrix with the specified properties.
    """
    m, n = shape
    center_m, center_n = center

    # Create a grid of indices representing the row and column positions
    rows_indices, cols_indices = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')

    # Calculate the distance of each element from the center
    distances = np.sqrt((rows_indices - center_m) ** 2 + (cols_indices - center_n) ** 2)

    # Create the Gaussian matrix using the formula of the Gaussian distribution
    gaussian = np.exp(-distances ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    if normalized:
        gaussian_max, gaussian_min = gaussian.max(), gaussian.min()
        gaussian = (gaussian - gaussian_min) / (gaussian_max - gaussian_min)

    return gaussian

class GazeAttentionMatrix():

    def __init__(self, device):

        self.device = device
        self.sigma = 60

        self.maximum_image_shape = None
        self._filter_size = None
        self._filter_map_center_location = None
        self.filter_map = None

        self.gaze_attention_pixel_map_buffer = None

    def set_maximum_image_shape(self, image_shape):
        self.maximum_image_shape = image_shape

        self._filter_size = self.maximum_image_shape * 2 - 1
        self._filter_map_center_location = self.maximum_image_shape - 1

        self._filter_map = torch.tensor(
            gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=self.sigma,
                            normalized=True), device=self.device)

    def get_gaze_on_image_attention_map(self, attention_center_location, image_shape):

        x_offset_min = self._filter_map_center_location[0] - attention_center_location[0]
        x_offset_max = x_offset_min + image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - attention_center_location[1]
        y_offset_max = y_offset_min + image_shape[1]

        gaze_on_image_attention_map = self._filter_map[x_offset_min: x_offset_max,
                                      y_offset_min:y_offset_max].clone()  # this is a copy!!!

        return gaze_on_image_attention_map

    #
    def gaze_attention_pixel_map_clutter_removal(self, gaze_on_grid_attention_map, attention_clutter_ratio=0.1):
        self.gaze_attention_pixel_map_buffer = attention_clutter_ratio * self.gaze_attention_pixel_map_buffer + (
                1 - attention_clutter_ratio) * gaze_on_grid_attention_map

def fix_seq_to_gaze_map(fix_seq, original_image_size, max_image_size=(3000, 6000)):
    current_image_shape = np.array(original_image_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gaze_attention_matrix = GazeAttentionMatrix(device=device)

    gaze_attention_matrix.set_maximum_image_shape(np.array(max_image_size))
    gaze_attention_matrix.gaze_attention_pixel_map_buffer = torch.tensor(np.zeros(shape=current_image_shape), device=device)

    for fixation_point in fix_seq:
        gaze_on_image_attention_map = gaze_attention_matrix.get_gaze_on_image_attention_map(fixation_point, current_image_shape)
        gaze_attention_matrix.gaze_attention_pixel_map_clutter_removal(gaze_on_image_attention_map, attention_clutter_ratio=0.995)
    gaze_attention_map = gaze_attention_matrix.gaze_attention_pixel_map_buffer.detach().cpu().numpy()
    return gaze_attention_map