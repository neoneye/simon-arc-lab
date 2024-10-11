# Mask of the half plane.
# - The halfplane is defined by two points, with different directionality across the input/output pairs.
# - The halfplane is adjacent to a single point, with the same directionality across the input/output pairs.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_mask import *
from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_halfplane import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_halfplane'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_halfplane.jsonl')

def generate_task_halfplane_with_two_pixels(seed: int) -> Task:
    """
    Show 2 pixels, and draw a half plane adjacent to them.
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 5
    max_image_size = 12

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)
    color_map_input = {}
    for i in range(10):
        color_map_input[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    color_map_output = {}
    for i in range(10):
        color_map_output[i] = output_colors[i]

    task.metadata_task_id = 'halfplane_with_two_pixels'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            # number_of_positions = random.Random(iteration_seed + 1).randint(1, 3)
            number_of_positions = 2

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            positions = []
            for i in range(number_of_positions):
                x = random.Random(iteration_seed + 3 + i * 2).randint(0, width - 1)
                y = random.Random(iteration_seed + 4 + i * 2).randint(0, height - 1)
                xy = (x, y)
                if xy in positions:
                    continue
                positions.append(xy)
            if len(positions) != 2:
                continue
            x0, y0 = positions[0]
            x1, y1 = positions[1]

            background_image = image_create(width, height, 0)
            input_image_raw = background_image.copy()
            input_image_raw[y0, x0] = 1
            input_image_raw[y1, x1] = 2

            accumulated_mask = background_image.copy()
            accumulated_mask = image_halfplane(accumulated_mask, x0, y0, x1, y1)
            accumulated_mask[y0, x0] = 2
            accumulated_mask[y1, x1] = 2
 
            input_image = image_replace_colors(input_image_raw, color_map_input)
            output_image = image_replace_colors(accumulated_mask, color_map_output)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_halfplane_with_one_pixel(seed: int) -> Task:
    """
    Show 1 pixel, and draw a half plane adjacent to it.
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 5
    max_image_size = 12

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)
    color_map_input = {}
    for i in range(10):
        color_map_input[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    color_map_output = {}
    for i in range(10):
        color_map_output[i] = output_colors[i]

    dx = None
    dy = None
    name = None
    variant_index = seed % 8
    dx_dy_name_list = [
        (10, 0, 'top'),
        (-10, 0, 'bottom'),
        (0, -10, 'left'),
        (0, 10, 'right'),
        (10, 10, 'topright'),
        (-10, 10, 'bottomright'),
        (10, -10, 'topleft'),
        (-10, -10, 'bottomleft'),
    ]
    dx, dy, name = dx_dy_name_list[variant_index]

    task.metadata_task_id = f'halfplane_with_one_pixel_{name}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            x0 = random.Random(iteration_seed + 3 + i * 2).randint(0, width - 1)
            y0 = random.Random(iteration_seed + 4 + i * 2).randint(0, height - 1)
            x1 = x0 + dx
            y1 = y0 + dy

            background_image = image_create(width, height, 0)
            input_image_raw = background_image.copy()
            input_image_raw[y0, x0] = 1

            output_image_raw = background_image.copy()
            output_image_raw = image_halfplane(output_image_raw, x0, y0, x1, y1)
            output_image_raw[y0, x0] = 2
 
            input_image = image_replace_colors(input_image_raw, color_map_input)
            output_image = image_replace_colors(output_image_raw, color_map_output)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 2
    if j == 0:
        task = generate_task_halfplane_with_one_pixel(seed)
    elif j == 1:
        task = generate_task_halfplane_with_two_pixels(seed)
    transformation_id = task.metadata_task_id
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=122155117,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
