# Based on plot_xyt(), plot_single_image() by Minseo Kim
# https://www.kaggle.com/code/minseo14/arc-task-00d62c1b-with-cnn
from typing import Optional
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

def plot_single_image(image: np.array, ax, title: str, cmap, norm, show_grid: bool):
    height, width = image.shape

    ax.imshow(image, cmap=cmap, norm=norm)
    if show_grid:
        ax.grid(True, which='both', color=GRID_COLOR, linewidth = 1)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(image[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(image))])

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    
    ax.set_title(title, fontweight='bold')

    label = f"{width}x{height}"
    label_params = {
        'fontsize': 12,
        'fontweight': 'normal',
        'color': LABEL_COLOR_WIDTHXHEIGHT
    }
    ax.set_xlabel(label, **label_params)

def show_prediction_result(input_image: np.array, predicted_image: np.array, expected_image: Optional[np.array], title: str = 'Task', show_grid: bool = True, save_path: Optional[str] = None):
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
    norm = colors.Normalize(vmin=0, vmax=9)
    
    plot_single_image(input_image, axs[0], 'Input', cmap, norm, show_grid)
    plot_single_image(predicted_image, axs[1], 'Predicted', cmap, norm, show_grid)
    if expected_image is not None:
        plot_single_image(expected_image, axs[2], 'Expected', cmap, norm, show_grid)
    
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)

    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith('.png'):
            raise ValueError('save_path must end with .png')    
        plt.savefig(save_path)
        plt.close()
        print('saved to: {}'.format(save_path))
    else:
        plt.show()

