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
IMAGE_BORDER_COLOR = GRID_COLOR
LABEL_COLOR_TRAIN_PAIR = '#444'
LABEL_COLOR_TEST_PAIR = '#222'
PLOT_BACKGROUND_COLOR = '#dddddd'

def plot_single_image(image: np.array, ax, title: str, cmap, norm):
    height, width = image.shape

    ax.imshow(image, cmap=cmap, norm=norm)
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
        'color': LABEL_COLOR_TRAIN_PAIR
    }

    ax.set_xlabel(label, **label_params)

def plot_xyt(input_image: np.array, predicted_image: np.array, expected_image: Optional[np.array], title: str):
    """Plots the input, predicted, and answer pairs of a specified task, using the ARC color scheme."""
    num_img = 3
    fig, axs = plt.subplots(1, num_img, figsize=(9, num_img))
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(ARCAGI_COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)
    
    plot_single_image(input_image, axs[0], 'Input', cmap, norm)
    plot_single_image(predicted_image, axs[1], 'Predicted', cmap, norm)
    if expected_image is not None:
        plot_single_image(expected_image, axs[2], 'Answer', cmap, norm)
    
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)

    fig.tight_layout()
    plt.show()

