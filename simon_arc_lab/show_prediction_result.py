# Based on plot_xyt(), plot_single_image() by Minseo Kim
# https://www.kaggle.com/code/minseo14/arc-task-00d62c1b-with-cnn
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

ARCAGI_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

GRID_COLOR = '#555555'
LABEL_COLOR_WIDTHXHEIGHT = '#444'
PLOT_BACKGROUND_COLOR = '#dddddd'

def plot_single_image(ax, title: str, image: Optional[np.array], cmap, norm, show_grid: bool, mask_outside09: bool):
    if image is None:
        plot_missing_image(ax, title)
        return

    if mask_outside09:
        # Mask values outside 0-9, so they can be shown with the cmap bad color
        image = np.ma.masked_where((image < 0) | (image > 9), image)

    height, width = image.shape

    ax.imshow(image, cmap=cmap, norm=norm)
    if show_grid:
        ax.grid(True, which='both', color=GRID_COLOR, linewidth = 1)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + width)])
    ax.set_yticks([x-0.5 for x in range(1 + height)])

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    
    ax.set_title(title, fontweight='bold')

    label = f"{width}x{height}"
    label_params = {
        'fontsize': 12,
        'fontweight': 'normal',
        'color': LABEL_COLOR_WIDTHXHEIGHT
    }
    ax.set_xlabel(label, **label_params)

def plot_missing_image(ax, title: str):
    width = 1
    height = 1
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + width)])
    ax.set_yticks([x-0.5 for x in range(1 + height)])

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    
    ax.set_title(title, fontweight='bold')

def show_prediction_result(input_image: np.array, predicted_image: Optional[np.array], expected_image: Optional[np.array], title: str = 'Task', show_grid: bool = True, save_path: Optional[str] = None):
    """
    Plots the input, predicted, and answer pairs of a specified task, using the ARC color scheme.

    input_image: The input image.
    predicted_image: The predicted image.
    expected_image: The expected image.
    title: The title of the plot.
    show_grid: Whether to show the grid lines.
    save_path: The path to save the plot as a PNG file. If None, the plot will be displayed.
    """
    num_img = 3
    fig, axs = plt.subplots(1, num_img, figsize=(9, num_img))
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(ARCAGI_COLORS)
    cmap.set_bad(color='white')
    norm = colors.Normalize(vmin=0, vmax=9)
    
    plot_single_image(axs[0], 'Input', input_image, cmap, norm, show_grid, mask_outside09=True)
    plot_single_image(axs[1], 'Predicted', predicted_image, cmap, norm, show_grid, mask_outside09=True)
    plot_single_image(axs[2], 'Expected', expected_image, cmap, norm, show_grid, mask_outside09=True)
    
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)

    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith('.png'):
            raise ValueError('save_path must end with .png')    
        plt.savefig(save_path)
        plt.close()
        # print('saved to: {}'.format(save_path))
    else:
        plt.show()

def show_multiple_images(cmap_title_image_list: list[Tuple[str, np.array, str]], title: str = 'Task', show_grid: bool = True, save_path: Optional[str] = None):
    """
    Plots the multiple ARC images side by side.

    cmap_title_image_list: List of tuples with image colormap, image title and image data.
    title: The title of the plot.
    show_grid: Whether to show the grid lines.
    save_path: The path to save the plot as a PNG file. If None, the plot will be displayed.
    """
    num_img = len(cmap_title_image_list)
    resolution = 20
    fig, axs = plt.subplots(1, num_img, figsize=(resolution, (resolution / (num_img + 0.5)) + 1))
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(ARCAGI_COLORS)
    cmap.set_bad(color='white')
    norm09 = colors.Normalize(vmin=0, vmax=9)
    norm01 = colors.Normalize(vmin=0, vmax=1, clip=True)
    
    for index, (image_cmap, image_title, image_data) in enumerate(cmap_title_image_list):
        if image_cmap == 'arc':
            plot_single_image(axs[index], image_title, image_data, cmap, norm09, show_grid, mask_outside09=True)
        elif image_cmap == 'heatmap':
            plot_single_image(axs[index], image_title, image_data, 'bone', norm01, show_grid, mask_outside09=False)
        else:
            raise ValueError('image_cmap must be either "arc" or "heatmap"')
    
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)

    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith('.png'):
            raise ValueError('save_path must end with .png')    
        plt.savefig(save_path)
        plt.close()
        # print('saved to: {}'.format(save_path))
    else:
        plt.show()

