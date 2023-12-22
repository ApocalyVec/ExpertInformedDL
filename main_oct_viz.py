from eidl.viz.viz_oct_results import viz_oct_results

results_dir = 'results-12_19_2023_23_46_06'

batch_size = 8

viz_val_acc = True

if __name__ == '__main__':
    viz_oct_results(results_dir, batch_size, viz_val_acc=viz_val_acc, plot_format='individual')