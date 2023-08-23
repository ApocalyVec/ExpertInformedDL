# train parameters for train.py ########################################
# train parameters
# data_root = "/media/rena-researcher/Elements/ExpertViT/Datasets/ODIR"
data_root = 'D:/Dropbox/Dropbox/ExpertViT/Datasets/ODIR'
# data_root = 'C:/Users/S-Vec/Dropbox/ExpertViT/Datasets/ODIR'

# save_dir = 'SavedModels/ViT/'
# model_name = 'ODIRViT'

# save_dir = 'SavedModels/ResNet/'
# model_name = 'ODIRResNet'



train_ratio = 0.85
batch_size = 12
epochs = 1
lr = 1e-2

normalize_using_dataset_values = False
normalize_mean_resnet = [0.485, 0.456, 0.406]
normalize_std_resnet = [0.229, 0.224, 0.225]
normalize_mean_ViTPretrained = [0.5, 0.5, 0.5]
normalize_std_ViTPretrained = [0.5, 0.5, 0.5]

image_size_VitPretrained = [384, 384]
# model hyperparameters  ########################################

# VIT hyperparameters
vit_depth = 6

# evaluation parameterse for eval.py ########################################
results_dir = "results-08_15_2023_15_43_28/SavedModels/ViT/11-16-2022/training_histories.pickle'"