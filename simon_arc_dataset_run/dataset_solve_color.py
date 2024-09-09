# color transformations:
# - Swap 2 colors.
# - Identify the most popular color.
# - Identify the least popular color.
#
# IDEA: Mask of where a particular color occurs in the input.
# IDEA: Replace one color with another color.
#
# IDEA: Swap the most/lest popular colors with each other.
#
# IDEA: Image size 1xN, where N is the number of unique colors.
# IDEA: Image size 1xN, where N is the count of the most popular color.
# IDEA: Image size 1xN, where N is the count of the least popular color.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_pad import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_color'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_color.jsonl')

def generate_task_replace_color(seed: int, transformation_id: str) -> Task:
    """
    Replace one color with another color.

    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b1948b0a
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 6
    min_padding = 1
    max_padding = 5

    color_padding = 0
    color_background = 1
    color_replace_from = 2
    color_replace_to = 3

    color_map_replace = {
        color_replace_from: color_replace_to,
    }

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    task.metadata_task_id = f'replace_color {transformation_id}'

    for i in range(count_example+count_test):
        is_example = i < count_example

        mask_image = None
        for retry_index in range(10):
            iteration_seed = seed + 1000 + retry_index * 100033 + i
            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, color_background, color_replace_from, ratio, iteration_seed + 4)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() == 2:
                # print(f"retry_index: {retry_index}")
                break

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")
        
        mask_image_with_padding = image_pad_random(mask_image, seed + 1000 + i * 133, color_padding, min_padding, max_padding)

        if transformation_id == 'no_padding':
            input_image_raw = mask_image.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'crop':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'padding':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image_with_padding.copy()
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")

        output_image_with_replaced_colors = image_replace_colors(output_image_raw, color_map_replace)
        input_image = image_replace_colors(input_image_raw, color_map)
        output_image = image_replace_colors(output_image_with_replaced_colors, color_map)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_swap_colors(seed: int) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 1).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'swap_colors'
    min_width = 1
    max_width = 14
    min_height = 1
    max_height = 14

    for i in range(count_example+count_test):
        is_example = i < count_example

        mask_image = None
        for retry_index in range(10):
            iteration_seed = seed + 1000 + i
            use_min_width = min_width
            use_min_height = min_height
            if retry_index == 1:
                use_min_width = 2
                use_min_height = 2
            if retry_index >= 2:
                use_min_width = 3
                use_min_height = 3
            width = random.Random(iteration_seed + 1).randint(use_min_width, max_width)
            height = random.Random(iteration_seed + 2).randint(use_min_height, max_height)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, iteration_seed + 4)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() == 2:
                # print(f"retry_index: {retry_index}")
                break

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")

        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(seed + 1000 + i).shuffle(colors)
        color0 = colors[0]
        color1 = colors[1]

        color_map = {
            0: color0,
            1: color1,
        }
        color_map_swapped = {
            0: color1,
            1: color0,
        }

        input_image = image_replace_colors(mask_image, color_map)
        output_image = image_replace_colors(mask_image, color_map_swapped)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_mostleast_popular_color(seed: int, find_id: str, output_size_id: str) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'mostleast_popular_color {find_id} {output_size_id}'
    min_width = 1
    max_width = 14
    min_height = 1
    max_height = 14

    for i in range(count_example+count_test):
        is_example = i < count_example

        random_image = None
        found_color = None
        number_of_retries = 0
        for retry_index in range(100):
            iteration_seed = seed + 1000 + i + retry_index * 100033
            use_min_width = min_width
            use_min_height = min_height
            if retry_index == 1:
                use_min_width = 2
                use_min_height = 2
            if retry_index >= 2:
                use_min_width = 3
                use_min_height = 3
            random_image = image_create_random_advanced(iteration_seed, use_min_width, max_width, use_min_height, max_height)
            histogram = Histogram.create_with_image(random_image)
            found_color = None
            if find_id == 'most_popular':
                found_color = histogram.most_popular_color()
            elif find_id == 'least_popular':
                found_color = histogram.least_popular_color()
            else:
                raise ValueError(f"Unknown find_id: {find_id}")
            
            if found_color is not None:
                number_of_retries = retry_index
                break

        if random_image is None:
            raise ValueError(f"Failed to create random image")
        if found_color is None:
            raise ValueError(f"Failed to find color")
        if number_of_retries >= 50:
            print(f"number_of_retries: {number_of_retries}")

        input_image = random_image

        output_width = None
        output_height = None
        if output_size_id == '1x1':
            output_width = 1
            output_height = 1
        elif output_size_id == 'same':
            output_width = input_image.shape[1]
            output_height = input_image.shape[0]
        else:
            raise ValueError(f"Unknown output_size_id: {output_size_id}")
        
        output_image = image_create(output_width, output_height, found_color)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 3
    # j = 0
    if j == 0:
        transformation_id = 'replace_color no_padding'
        task = generate_task_replace_color(seed, 'no_padding')
    elif j == 1:
        transformation_id = 'replace_color crop'
        task = generate_task_replace_color(seed, 'crop')
    elif j == 2:
        transformation_id = 'replace_color padding'
        task = generate_task_replace_color(seed, 'padding')
    elif j == 3:
        transformation_id = 'swap_colors'
        task = generate_task_swap_colors(seed)
    elif j == 4:
        transformation_id = 'most_popular_color_1x1'
        task = generate_task_mostleast_popular_color(seed, 'most_popular', '1x1')
    elif j == 5:
        transformation_id = 'least_popular_color_1x1'
        task = generate_task_mostleast_popular_color(seed, 'least_popular', '1x1')
    elif j == 6:
        transformation_id = 'most_popular_color_same'
        task = generate_task_mostleast_popular_color(seed, 'most_popular', 'same')
    elif j == 7:
        transformation_id = 'least_popular_color_same'
        task = generate_task_mostleast_popular_color(seed, 'least_popular', 'same')
    else:
        raise ValueError(f"Unknown j: {j}")
    
    # task.show()
    return generate_dataset_item_list_inner(seed, task, transformation_id)

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=13600232,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
