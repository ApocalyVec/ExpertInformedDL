import os

from eidl.viz.viz_oct_results import viz_oct_results

results_dir = '../temp/results-repaired-inception'
figure_dir = '../temp/results-repaired-inception/figures'

batch_size = 8

viz_val_acc = True

if __name__ == '__main__':
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)
    viz_oct_results(results_dir, batch_size, viz_val_acc=viz_val_acc, plot_format='individual', figure_dir=figure_dir)