# plot_task() by Minseo Kim
# https://www.kaggle.com/code/minseo14/arc-task-00d62c1b-with-cnn
from .task import Task
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines
import os

ARCAGI_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

GRID_COLOR = '#555555'

PAIR_LABEL_COLOR = '#202020'

def draw_grid_lines(ax, input_matrix, grid_color):
    height, width = input_matrix.shape
    # Draw horizontal lines
    for i in range(height + 1):
        x0 = -0.5
        x1 = width - 0.5
        y = i - 0.5
        line = mlines.Line2D([x0, x1], [y, y], color=grid_color, linewidth=0.5)
        ax.add_line(line)
    # Draw vertical lines
    for j in range(width + 1):
        x = j - 0.5
        y0 = -0.5
        y1 = height - 0.5
        line = mlines.Line2D([x, x], [y0, y1], color=grid_color, linewidth=0.5)
        ax.add_line(line)

def plot_task(dataset, idx, data_category, fdir_to_save=None):
    """Plots the train and test pairs of a specified task, using the ARC color scheme."""

    def plot_one(input_matrix, ax, train_or_test, input_or_output, cmap, norm):
        height, width = input_matrix.shape

        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        draw_grid_lines(ax, input_matrix, grid_color=GRID_COLOR)

         # Remove the axes spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([])
        ax.set_yticks([])

        size = f"{width}x{height}"

        if train_or_test == 'test':
            label = 'TEST ' + size
        else:
            label = size

        if input_or_output == 'output':
            ax.set_xlabel(label, fontweight='bold', color=PAIR_LABEL_COLOR)
        else:
            ax.set_title(label, fontweight='bold', color=PAIR_LABEL_COLOR)

    train_inputs, train_outputs, test_inputs, test_outputs, task_id = dataset[idx]  # Load the first task
    
    num_train = len(train_inputs)
    num_test = len(test_inputs)
    num_total = num_train + num_test
    
    fig, axs = plt.subplots(2, num_total, figsize=(3*num_total, 3*2))
    plt.suptitle(f'{data_category.capitalize()} Set #{idx+1}, {task_id}:', fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(ARCAGI_COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)

    for j in range(num_train):
        plot_one(train_inputs[j], axs[0, j], 'train', 'input', cmap, norm)
        plot_one(train_outputs[j], axs[1, j], 'train', 'output', cmap, norm)

    for j in range(num_test):
        plot_one(test_inputs[j], axs[0, j + num_train], 'test', 'input', cmap, norm)
        if test_outputs != []:
            plot_one(test_outputs[j], axs[1, j + num_train], 'test', 'output', cmap, norm)
        else:
            plot_one([[5]], axs[1, j + num_train], 'test', 'output', cmap, norm)

    fig.patch.set_facecolor('#dddddd')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    if fdir_to_save is not None:
        fname = os.path.join(fdir_to_save, '{}_{}.png'.format(idx+1, task_id))
        plt.savefig(fname)
        plt.close()
        print('{} saved'.format(fname))
    else:
        plt.show()


def task_show_matplotlib(task: Task, answer=True):

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
        output = task.test_input(i)
        test_outputs.append(output)

    dataset = []

    task_id = 'x'
    dataset_item = [train_inputs, train_outputs, test_inputs, test_outputs, task_id]
    dataset.append(dataset_item)
    dataset.append(dataset_item)

    plot_task(dataset, 1, 'y', fdir_to_save=None)
