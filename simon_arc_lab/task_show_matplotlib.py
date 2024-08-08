# plot_task() by Minseo Kim
# https://www.kaggle.com/code/minseo14/arc-task-00d62c1b-with-cnn
from .task import Task
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os

ARCAGI_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

GRID_COLOR = '#555555'
IMAGE_BORDER_COLOR = GRID_COLOR
LABEL_COLOR_TRAIN_PAIR = '#444'
LABEL_COLOR_TEST_PAIR = '#222'
TASK_BACKGROUND_COLOR = '#dddddd'

def plot_task(dataset, task_title, show_grid, fdir_to_save=None, save_filename='task'):
    """Plots the train and test pairs of a specified task, using the ARC color scheme."""

    def plot_one(input_matrix, ax, train_or_test, input_or_output, cmap, norm, show_grid: bool):
        height, width = input_matrix.shape

        ax.imshow(input_matrix, cmap=cmap, norm=norm)

        if show_grid:
            ax.grid(True, which='both', color=GRID_COLOR, linewidth = 1)
            # The grid lines isn’t perfectly aligned with the image pixels. 
            # The neighbor pixels can be seen on the other side of the grid lines.

        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_edgecolor(IMAGE_BORDER_COLOR)

        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + width)])
        ax.set_yticks([y-0.5 for y in range(1 + height)])

        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

        size = f"{width}x{height}"

        if train_or_test == 'test':
            label = 'TEST ' + size
        else:
            label = size

        label_params = {'fontsize': 12}
        if train_or_test == 'test':
            label_params['fontweight'] = 'bold'
            label_params['color'] = LABEL_COLOR_TEST_PAIR
        else:
            label_params['fontweight'] = 'normal'
            label_params['color'] = LABEL_COLOR_TRAIN_PAIR

        if input_or_output == 'output':
            ax.set_xlabel(label, **label_params)
        else:
            ax.set_title(label, **label_params)

    train_inputs, train_outputs, test_inputs, test_outputs = dataset[0]  # Load the first task
    
    num_train = len(train_inputs)
    num_test = len(test_inputs)
    num_total = num_train + num_test
    
    fig, axs = plt.subplots(2, num_total, figsize=(3*num_total, 3*2))
    plt.suptitle(task_title, fontsize=20, fontweight='bold', y=0.96, ha='left', x=0.02)
    
    cmap = colors.ListedColormap(ARCAGI_COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)

    for j in range(num_train):
        plot_one(train_inputs[j], axs[0, j], 'train', 'input', cmap, norm, show_grid)
        plot_one(train_outputs[j], axs[1, j], 'train', 'output', cmap, norm, show_grid)

    for j in range(num_test):
        plot_one(test_inputs[j], axs[0, j + num_train], 'test', 'input', cmap, norm, show_grid)
        if test_outputs[j] is None:
            image = np.zeros((0, 0), dtype=np.uint8)
            plot_one(image, axs[1, j + num_train], 'test', 'output', cmap, norm, False)
        else:
            plot_one(test_outputs[j], axs[1, j + num_train], 'test', 'output', cmap, norm, show_grid)

    fig.patch.set_facecolor(TASK_BACKGROUND_COLOR)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    if fdir_to_save is not None:
        fname = os.path.join(fdir_to_save, '{}.png'.format(save_filename))
        plt.savefig(fname)
        plt.close()
        print('{} saved'.format(fname))
    else:
        plt.show()


def task_show_matplotlib(task: Task, show_grid: bool, show_answer: bool):

    train_inputs = []
    for i in range(task.count_examples):
        input = task.example_input(i)
        train_inputs.append(input)

    train_outputs = []
    for i in range(task.count_examples):
        input = task.example_output(i)
        train_outputs.append(input)

    test_inputs = []
    for i in range(task.count_tests):
        input = task.test_input(i)
        test_inputs.append(input)
    
    test_outputs = []
    for i in range(task.count_tests):
        if show_answer:
            output = task.test_output(i)
        else:
            output = None
        test_outputs.append(output)


    dataset_item = [train_inputs, train_outputs, test_inputs, test_outputs]
    dataset = [dataset_item]

    if task.metadata_task_id is None:
        task_title = 'Task without id'
    else:
        task_title = task.metadata_task_id

    fdir_to_save = None
    # fdir_to_save = '.'
    plot_task(dataset, task_title, show_grid, fdir_to_save)
