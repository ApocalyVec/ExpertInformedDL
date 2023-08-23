from source.viz.viz_oct_results import viz_oct_results

results_dir = 'results-08_15_2023_19_57_21'

batch_size = 8

viz_val_acc = True

if __name__ == '__main__':
    viz_oct_results(results_dir, batch_size, viz_val_acc=viz_val_acc, plot_format='individual', rollout_alpha=1.0)