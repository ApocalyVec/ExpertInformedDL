import sys
import os
import pandas as pd


def load_oct_data(data_dir, expert_data_root='ExpertData', experiment_image_root='ExpIms'):
    expert_data_root = os.path.join(data_dir, expert_data_root)
    experiment_image_root = os.path.join(data_dir, experiment_image_root)

    for experiment_days in os.listdir(expert_data_root):  # iterate over the days
        experiment_days_dir = os.path.join(expert_data_root, experiment_days)

        for experiment in os.listdir(experiment_days_dir):
            experiment_dir = os.path.join(experiment_days_dir, experiment)
            for expert in os.listdir(experiment_dir):
                expert_dir = os.path.join(experiment_dir, expert)
                for image_name in os.listdir(expert_dir):
                    image_dir = os.path.join(expert_dir, image_name)
                    image_path = os.path.join(experiment_image_root, image_name)
                    for fixation in os.listdir(image_dir):
                        fixation_dir = os.path.join(image_dir, fixation)
                        fixation_path = os.path.join(image_path, fixation)
                        df = pd.read_csv(fixation_dir)
                        sequences = np.array(df[['norm_pos_x', 'norm_pos_y']])
                        sequences[:, 1] = 1 - sequences[:, 1]
                        yield image_path, sequences



if __name__ == '__main__':
    data_root = 'D:/Dropbox/ExpertViT/Datasets/OCTData'

