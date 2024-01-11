import os
import pickle

import cv2
import imageio
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from PIL import Image

from eidl.utils.image_utils import remap_subimage_aoi, process_aoi
from eidl.utils.iter_utils import chunker, collate_fn
from eidl.utils.model_utils import parse_model_parameter, get_best_model, parse_training_results
from eidl.utils.torch_utils import any_image_to_tensor
from eidl.viz.vit_rollout import VITAttentionRollout

from eidl.viz.viz_utils import plt2arr, plot_train_history, plot_subimage_rolls, plot_image_attention


def register_cmap_with_alpha(cmap_name):
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))
    # change alpha values
    color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)
    # create a colormap object
    cmap_rtn = f'{cmap_name}_alpha'
    map_object = LinearSegmentedColormap.from_list(name=cmap_rtn, colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    return cmap_rtn

def viz_oct_results(results_dir, batch_size, n_jobs=1, acc_min=.3, acc_max=1, viz_val_acc=True, plot_format='individual', num_plot=14,
                    rollout_transparency=0.75, figure_dir=None):
    '''

    Parameters
    ----------
    results_dir
    test_image_path
    test_image_main
    batch_size
    image_size
    n_jobs
    acc_min
    acc_max
    viz_val_acc
    plot_format: can be 'individual' or 'grid'. Note setting to 'grid' will not plot the gifs
    num_plot

    Returns
    -------

    '''

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    image_stats = pickle.load(open(os.path.join(results_dir, 'image_stats.p'), 'rb'))
    # load the test dataset ############################################################################################
    test_dataset = pickle.load(open(os.path.join(results_dir, 'test_dataset.p'), 'rb'))
    results_dict, model_config_strings = parse_training_results(results_dir)

    # results_df.to_csv(os.path.join(results_dir, "summary.csv"))

    # visualize the val acc across alpha ###############################################################################
    alphas = {parse_model_parameter(x, 'alpha') for x in model_config_strings}
    alphas = list(alphas)
    alphas.sort()
    models = {parse_model_parameter(x, 'model') for x in model_config_strings}

    small_font_size = 24
    medium_font_size = 26
    large_font_size = 30

    plt.rc('font', size=small_font_size)
    plt.rc('axes', titlesize=small_font_size)
    plt.rc('axes', labelsize=small_font_size)
    plt.rc('xtick', labelsize=small_font_size)
    plt.rc('ytick', labelsize=small_font_size)
    plt.rc('legend', fontsize=small_font_size)
    plt.rc('figure', titlesize=large_font_size)

    if viz_val_acc:
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        xticks = np.array(list(range(1, len(alphas) + 1)))
        model_x_offset = 0.3
        box_width = 0.25
        colors = matplotlib.cm.tab20(range(20))

        for i, model in enumerate(models):
            val_accs = []
            for alpha in alphas:
                val_acc_alpha = []
                for model_config_string, results in results_dict.items():
                    if parse_model_parameter(model_config_string, 'alpha') == alpha and parse_model_parameter(model_config_string, 'model') == model:
                        val_acc_alpha.append(np.max(results['val_accs']))
                val_accs.append(val_acc_alpha)
            x_positions = xticks + model_x_offset * i
            plt.boxplot(val_accs, positions=x_positions, patch_artist=True, widths=box_width, boxprops=dict(facecolor=colors[i*2+1], alpha=0.5, color=colors[i*2]), whiskerprops=dict(color=colors[i*2]), capprops=dict(color=colors[i*2]), medianprops=dict(color=colors[i*2]))
            plt.plot(x_positions, [np.mean(x) for x in val_accs], label=f"{model} average across tested parameters", color=colors[i*2])
            plt.scatter(x_positions, [np.mean(x) for x in val_accs], color=colors[i*2], s=40)

        plt.ylim(acc_min, acc_max)
        plt.xticks(ticks=xticks, labels=alphas)
        plt.xlabel("Expert AOI weight (α)")
        plt.ylabel("Validation accuracy")
        plt.title(f"validation accuracy across expert AOI weights")
        plt.legend()
        plt.tight_layout()
        if figure_dir is not None:
            plt.savefig(os.path.join(figure_dir, f"validation accuracy across expert AOI weights.png"))
        plt.show()

        # visualize the hyperparam space ##################################################################################
        parameter_to_test_base = 'lr', 'depth', 'dist'
        parameter_to_test_pretrained = 'lr', 'dist'
        xticks = np.array(list(range(len(alphas))))
        for i, model in enumerate(models):
            for hyperparam_name in parameter_to_test_base if model == 'base' else parameter_to_test_pretrained:
                hyperparam_space = {parse_model_parameter(x, hyperparam_name) for x in model_config_strings if model == parse_model_parameter(x, 'model')}
                hyperparam_space = list(hyperparam_space)
                if isinstance(hyperparam_space[0], float):
                    hyperparam_space.sort()
                fig = plt.figure(figsize=(15, 10), constrained_layout=True)
                val_accs = np.empty((len(hyperparam_space), len(alphas)))
                for j, hyper_param in enumerate(hyperparam_space):
                    for k, alpha in enumerate(alphas):
                        val_acc_hyperparam_alpha = []
                        for model_config_string, results in results_dict.items():
                            if parse_model_parameter(model_config_string, hyperparam_name) == hyper_param and parse_model_parameter(model_config_string, 'alpha') == alpha and parse_model_parameter(model_config_string, 'model') == model:
                                val_acc_hyperparam_alpha.append(np.max(results['val_accs']))
                        val_accs[j, k] = np.mean(val_acc_hyperparam_alpha)
                        plt.text(k, j, round(float(np.mean(val_acc_hyperparam_alpha)), 3))
                plt.imshow(val_accs, vmin=acc_min, vmax=acc_max)
                plt.xticks(ticks=xticks, labels=alphas)
                plt.yticks(ticks=list(range(len(hyperparam_space))), labels=[float('%.1g' % x) if isinstance(x, float) else x for x in hyperparam_space ])  # additional float casting to avoid e notation
                plt.xlabel("Expert AOI weight (α)")
                plt.ylabel(hyperparam_name)
                plt.colorbar()
                plt.title(f"{model}: validation accuracy for {hyperparam_name}-alpha ")
                if figure_dir is not None:
                    plt.savefig(os.path.join(figure_dir, f"{model}: validation accuracy for {hyperparam_name}.png"))
                plt.show()

    # visualize the attention rollout ##################################################################################
    cmap_name = register_cmap_with_alpha('viridis')

    models = list(reversed(list(models)))

    best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)
    best_model.eval()
    patch_size = best_model.patch_height, best_model.patch_width
    model_depth = best_model.depth

    # visualize the training history of the best model ##################################################################
    plot_train_history(best_model_results, note=f"{best_model_config_string}")

    # register target cmap #########################################################
    has_subimage = test_dataset.trial_samples[0].keys()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # one image at a time

    roll_image_folder = os.path.join(figure_dir, 'rollout_images')
    if not os.path.isdir(roll_image_folder):
        os.mkdir(roll_image_folder)

    with torch.no_grad():
        test_dataset.create_aoi(best_model.get_grid_size())
        # epoch_loss, epoch_acc = run_validation(best_model, test_loader, device)
        # print(f"Test acc: {epoch_acc}")

        # use gradcam is model is not a ViT
        vit_rollout = VITAttentionRollout(best_model, device=device, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.1)
        sample_count = 0

        if plot_format == 'grid':
            fig, axs = plt.subplots(model_depth + 2, num_plot, figsize=(2 * num_plot, 2 * (model_depth + 2)))
            plt.setp(axs, xticks=[], yticks=[])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
            fig.tight_layout()

        for batch in test_loader:
            print(f'Processing sample {sample_count} in test set')
            image, label, label_encoded, fix_sequence, aoi_heatmap, image_resized, image_original, *rest = batch
            if has_subimage:
                # take out the batches
                subimage_positions = [x[0] for x in rest[0]]
                subimage_masks = [x[0].detach().cpu().numpy() for x in image['masks']]  # the masks for the subimages in a a single image
                subimages = [x[0].detach().cpu().numpy() for x in image['subimages']]  # the subimages in a single image
            else:
                subimage_masks = None
                subimage_positions = None

            rolls = []
            fixation_sequence_torch = torch.Tensor(rnn_utils.pad_sequence(fix_sequence, batch_first=True))

            for roll_depth in range(best_model.depth):
                image = any_image_to_tensor(image, device)
                rolls.append(vit_rollout(depth=roll_depth, in_data=image, fixation_sequence=fixation_sequence_torch))

            image_original = np.array(image_original[0].numpy(), dtype=np.uint8)
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
            image_original_size = image_original.shape[:2]
            if plot_format == 'individual':
                fig_list = []
                # plot the original image
                fig = plt.figure(figsize=(15, 10), constrained_layout=True)
                plt.imshow(image_original)  # plot the original image, bgr to rgb
                plt.axis('off')
                plt.title(f'#{sample_count}, original image')
                if figure_dir is not None:
                    Image.fromarray(image_original).save(os.path.join(figure_dir, f'#{sample_count}, original image.png'))
                plt.show()
                fig_list.append(plt2arr(fig))
                # plot the original image with expert AOI heatmap
                fig = plt.figure(figsize=(15, 10), constrained_layout=True)
                plt.imshow(image_original)  # plot the original image

                _aoi_heatmap, *_ = process_aoi(aoi_heatmap[0].numpy(), image_original_size, has_subimage, grid_size=best_model.get_grid_size(),
                                           subimage_masks=subimage_masks, subimages=subimages, subimage_positions=subimage_positions, patch_size=patch_size)
                plt.imshow(_aoi_heatmap, cmap=cmap_name, alpha=rollout_transparency)
                plt.axis('off')
                plt.title(f'#{sample_count}, expert AOI')
                plt.colorbar()
                if figure_dir is not None:
                    plt.savefig(os.path.join(figure_dir, f'#{sample_count}, expert AOI.png'))
                plt.show()

                for i, roll in enumerate(rolls):
                    rollout_image, subimage_roll = process_aoi(rolls[0], image_original_size, has_subimage,
                                               grid_size=best_model.get_grid_size(),
                                               subimage_masks=subimage_masks, subimages=subimages, subimage_positions=subimage_positions, patch_size=patch_size)

                    plot_image_attention(image_original, rollout_image, _aoi_heatmap, cmap_name='plasma',
                                         notes=f'#{sample_count} model {model}, roll depth {i}', save_dir=roll_image_folder)
                    plot_subimage_rolls(subimage_roll, subimages, subimage_positions, image_stats['image_stds'], image_stats['image_means'], cmap_name='plasma',
                                        notes=f"#{sample_count} model: {model}, roll depth {i}", overlay_alpha=rollout_transparency, save_dir=roll_image_folder)

                    # fig.savefig(f'figures/valImageIndex-{sample_count}_model-{model}_rollDepth-{i}.png')
                    fig_list.append(plt2arr(fig))
                # imageio.mimsave(f'gifs/model-{model}_valImageIndex-{sample_count}.gif', fig_list, fps=2)  # TODO expose save dir
            elif plot_format == 'grid' and sample_count < num_plot:
                    axis_original_image, axis_aoi_heatmap, axes_roll = axs[0, sample_count], axs[1, sample_count], axs[2:, sample_count]
                    axis_original_image.imshow(image_original)  # plot the original image
                    axis_original_image.axis('off')
                    # axis_original_image.title(f'#{sample_count}, original image')

                    # plot the original image with expert AOI heatmap
                    axis_aoi_heatmap.imshow(image_original)  # plot the original image
                    _aoi_heatmap = cv2.resize(aoi_heatmap.numpy(), dsize=image.shape[1:], interpolation=cv2.INTER_LANCZOS4)
                    axis_aoi_heatmap.imshow(_aoi_heatmap.T, cmap=cmap_name, alpha=rollout_transparency)
                    axis_aoi_heatmap.axis('off')
                    # axis_aoi_heatmap.title(f'#{sample_count}, expert AOI')

                    for i, roll in enumerate(rolls):
                        rollout_image = cv2.resize(roll, dsize=image.shape[1:], interpolation=cv2.INTER_LANCZOS4)
                        axes_roll[i].imshow(np.moveaxis(image_resized, 0, 2))  # plot the original image
                        axes_roll[i].imshow(rollout_image.T, cmap=cmap_name, alpha=rollout_transparency)
                        axes_roll[i].axis('off')
                        # axes_roll[i].title(f'#{sample_count}, model {model}, , roll depth {i}')
            sample_count += 1

    if plot_format == 'grid':
        plt.show()



