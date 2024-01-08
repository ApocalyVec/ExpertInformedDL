from eidl.viz.viz_oct_results import viz_oct_results

results_dir = '../temp/results-01_07_2024_10_53_56'

batch_size = 8

viz_val_acc = True

if __name__ == '__main__':
    viz_oct_results(results_dir, batch_size, viz_val_acc=viz_val_acc, plot_format='individual')