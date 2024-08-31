# Apply gravity to the input images in various directions.
# - row-wise gravity
# - column-wise gravity
#
# IDEA: Gravity with objects
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5ffb2104
#
# IDEA: Gravity with multiple non-moving objects
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=05f2a901
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9c56f360
#
# IDEA: Gravity with objects towards an attractor
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6ad5bdfd
#
# IDEA: Gravity following a ruler
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=98cf29f8
#
# IDEA: Alignment of objects
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=67636eac
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1caeab9d
#
# IDEA: Gravity into a hole with a particular shape
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=67c52801
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
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.image_rect import image_rect, image_rect_hollow
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_gravity_move import *
from simon_arc_lab.image_gravity_draw import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_gravity'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_gravity.jsonl')

def generate_task_gravity_move(seed: int, direction: GravityMoveDirection) -> Task:
    """
    Show a few lonely pixels, and apply gravity move.

    Examples:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1e0a9b12
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 10
    max_number_of_positions = 7

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)

    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    task.metadata_task_id = f'gravity_move {direction.name.lower()}'

    use_two_colors = random.Random(seed + 4).randint(0, 1) == 1

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            number_of_positions = random.Random(iteration_seed + 1).randint(2, max_number_of_positions)

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

            background_image = image_create(width, height, 0)
            input_image_raw = background_image.copy()
            for x, y in positions:
                if use_two_colors:
                    color = 1
                else:
                    color = random.Random(iteration_seed + 5 + x + y).randint(1, 9)
                input_image_raw[y, x] = color

            output_image_raw = image_gravity_move(input_image_raw, 0, direction)

            # We are not interested in images with zero lonely pixels
            histogram = Histogram.create_with_image(output_image_raw)
            if histogram.number_of_unique_colors() < 2:
                continue

            # if input and output are the same, we are not interested
            if np.array_equal(input_image_raw, output_image_raw):
                continue

            input_image = image_replace_colors(input_image_raw, color_map)
            output_image = image_replace_colors(output_image_raw, color_map)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_gravity_draw(seed: int, transformation_id: str) -> Task:
    """
    Show a few lonely pixels, and apply gravity draw.

    Examples:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=673ef223
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=82819916
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=90f3ed37
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=97999447
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=99fa7670
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 30
    max_number_of_positions = 7

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)

    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    task.metadata_task_id = f'gravity_draw {transformation_id}'

    if transformation_id == 'down':
        direction = GravityDrawDirection.DOWN
    elif transformation_id == 'up':
        direction = GravityDrawDirection.UP
    elif transformation_id == 'left':
        direction = GravityDrawDirection.LEFT
    elif transformation_id == 'right':
        direction = GravityDrawDirection.RIGHT
    else:
        raise Exception(f"Unknown transformation_id: {transformation_id}")

    use_two_colors = random.Random(seed + 4).randint(0, 1) == 1

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            number_of_positions = random.Random(iteration_seed + 1).randint(2, max_number_of_positions)

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

            background_image = image_create(width, height, 0)
            input_image_raw = background_image.copy()
            for x, y in positions:
                if use_two_colors:
                    color = 1
                else:
                    color = random.Random(iteration_seed + 5 + x + y).randint(1, 9)
                input_image_raw[y, x] = color

            output_image_raw = image_gravity_draw(input_image_raw, 0, direction)

            # We are not interested in images with zero lonely pixels
            histogram = Histogram.create_with_image(output_image_raw)
            if histogram.number_of_unique_colors() < 2:
                continue

            # if input and output are the same, we are not interested
            if np.array_equal(input_image_raw, output_image_raw):
                continue

            input_image = image_replace_colors(input_image_raw, color_map)
            output_image = image_replace_colors(output_image_raw, color_map)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 12
    # j = seed % 4
    j = (seed % 4) + 4
    if j == 0:
        transformation_id = 'gravity_move_top_to_bottom'
        task = generate_task_gravity_move(seed, GravityMoveDirection.TOP_TO_BOTTOM)
    elif j == 1:
        transformation_id = 'gravity_move_bottom_to_top'
        task = generate_task_gravity_move(seed, GravityMoveDirection.BOTTOM_TO_TOP)
    elif j == 2:
        transformation_id = 'gravity_move_left_to_right'
        task = generate_task_gravity_move(seed, GravityMoveDirection.LEFT_TO_RIGHT)
    elif j == 3:
        transformation_id = 'gravity_move_right_to_left'
        task = generate_task_gravity_move(seed, GravityMoveDirection.RIGHT_TO_LEFT)
    elif j == 4:
        transformation_id = 'gravity_move_topleft_to_bottomright'
        task = generate_task_gravity_move(seed, GravityMoveDirection.TOPLEFT_TO_BOTTOMRIGHT)
    elif j == 5:
        transformation_id = 'gravity_move_bottomright_to_topleft'
        task = generate_task_gravity_move(seed, GravityMoveDirection.BOTTOMRIGHT_TO_TOPLEFT)
    elif j == 6:
        transformation_id = 'gravity_move_topright_to_bottomleft'
        task = generate_task_gravity_move(seed, GravityMoveDirection.TOPRIGHT_TO_BOTTOMLEFT)
    elif j == 7:
        transformation_id = 'gravity_move_bottomleft_to_topright'
        task = generate_task_gravity_move(seed, GravityMoveDirection.BOTTOMLEFT_TO_TOPRIGHT)
    elif j == 8:
        transformation_id = 'gravity_draw_up'
        task = generate_task_gravity_draw(seed, 'up')
    elif j == 9:
        transformation_id = 'gravity_draw_down'
        task = generate_task_gravity_draw(seed, 'down')
    elif j == 10:
        transformation_id = 'gravity_draw_left'
        task = generate_task_gravity_draw(seed, 'left')
    elif j == 11:
        transformation_id = 'gravity_draw_right'
        task = generate_task_gravity_draw(seed, 'right')
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=26000194,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
