# plot_task() by Minseo Kim
# https://www.kaggle.com/code/minseo14/arc-task-00d62c1b-with-cnn
from .task import Task
import matplotlib.pyplot as plt
from matplotlib import colors
import os

ARCAGI_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

def plot_task(dataset, idx, data_category, fdir_to_save=None):
    """Plots the train and test pairs of a specified task, using the ARC color scheme."""

    def plot_one(input_matrix, ax, train_or_test, input_or_output, cmap, norm):
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True, which='both', color='lightgrey', linewidth = 0.5)
        
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        
        if train_or_test == 'test' and input_or_output == 'output':
            ax.set_title('TEST OUTPUT', color='green', fontweight='bold')
        else:
            ax.set_title(train_or_test + ' ' + input_or_output, fontweight='bold')

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

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
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


def plot_single_image(matrix, ax, title, cmap, norm):
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(matrix))])
    
    ax.set_title(title, fontweight='bold')

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
