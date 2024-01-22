import os

from eidl.viz.viz_oct_results import viz_oct_results

# results_dir = '../temp/results-repaired-base-vit'
# figure_dir = '../temp/results-repaired-base-vit/figures-paper'

# results_dir = '../temp/results-repaired-inception'
# figure_dir = '../temp/results-repaired-inception/figures-paper'

# results_dir = '../temp/results-repaired-pretrained-vit'
# figure_dir = '../temp/results-repaired-pretrained-vit/figures-paper'

# results_dir = '../temp/results-repaired-pretrained-vit-10folds'
# figure_dir = '../temp/results-repaired-pretrained-vit-10folds/figures-paper'

# results_dir = '../temp/results-repaired-resnet'
# figure_dir = '../temp/results-repaired-resnet/figures-paper'

# results_dir = '../temp/results-repaired-vgg'
# figure_dir = '../temp/results-repaired-vgg/figures-paper'

results_dir = '../temp/results-repaired-base-vit-10folds'
figure_dir = '../temp/results-repaired-base-vit-10folds/figures-paper'

batch_size = 8

viz_val_acc = True

if __name__ == '__main__':
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)
    viz_oct_results(results_dir, batch_size, viz_val_acc=viz_val_acc, plot_format='individual', figure_dir=figure_dir)