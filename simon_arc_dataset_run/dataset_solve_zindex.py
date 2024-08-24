# Z-index transformations.
# - extract mask of primary rectangle
# - place 2 rectangles on top of each other, and then restore the obscured area.
#
# IDEA: draw the obscured rectangle on top. 
# IDEA: mask with the intersection rectangle between the 2 rectangles.
# IDEA: crop out the obscured rectangle.
# IDEA: swap colors of the 2 rectangles.
# IDEA: draw diagonal lines on the obscured area.
# IDEA: draw rectangles that are 45 degree rotated.
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
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.task import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.image_util import *
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_zindex'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_zindex.jsonl')

def generate_task_mask_of_primary_rectangle(seed: int) -> Task:
    """
    Random noisy background with two colors.
    Draw a rectangle on top of the background.
    The job is to identify the rectangle.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 6
    max_image_size = 15

    # input colors
    colors_input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors_input)
    color_mapping_input = {
        0: colors_input[0],
        1: colors_input[1],
        2: colors_input[2],
    }

    # output colors
    colors_output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(colors_output)
    color_mapping_output = {
        0: colors_output[0],
        1: colors_output[1],
    }

    task.metadata_task_id = f'mask_of_primary_rectangle'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_seed = (retry_index * 10000) + ((seed + 17) * 37) + 101 + i

            image_width = random.Random(input_seed + 1).randint(min_image_size, max_image_size)
            image_height = random.Random(input_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.1, 0.2, 0.3, 0.4]
            ratio = random.Random(input_seed + 3).choice(ratios)
            image = image_create_random_with_two_colors(image_width, image_height, 0, 1, ratio, input_seed + 4)

            layer0_width = random.Random(input_seed + 4).randint(1, image_width)
            layer0_height = random.Random(input_seed + 5).randint(1, image_height)
            layer0_x = random.Random(input_seed + 6).randint(0, image_width - layer0_width)
            layer0_y = random.Random(input_seed + 7).randint(0, image_height - layer0_height)
            layer0_rect = Rectangle(
                layer0_x,
                layer0_y,
                layer0_width,
                layer0_height
            )
            # print(f"layer0_rect: {layer0_rect}")
            layer0_image = image_rect(image, layer0_rect, 2)

            mask_image = image_create(image_width, image_height, 0)
            layer0_mask = image_rect(mask_image, layer0_rect, 1)

            input_image = image_replace_colors(layer0_image, color_mapping_input)
            output_image = image_replace_colors(layer0_mask, color_mapping_output)
            break
        if input_image is None or output_image is None:
            raise Exception("Failed to create a pair.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_mask_of_obscured_rectangle(seed: int) -> Task:
    """
    Random noisy background with two colors.
    Draw 2 rectangles that overlaps.
    The job is to identify the rectangle that is obscured and repair the obscured pixels.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 6
    max_image_size = 15

    # input colors
    colors_input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors_input)
    color_mapping_input = {
        0: colors_input[0],
        1: colors_input[1],
        2: colors_input[2],
        3: colors_input[3],
    }

    # output colors
    colors_output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(colors_output)
    color_mapping_output = {
        0: colors_output[0],
        1: colors_output[1],
    }

    task.metadata_task_id = f'mask_of_obscured_rectangle'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_seed = (retry_index * 10000) + ((seed + 17) * 37) + 101 + i

            image_width = random.Random(input_seed + 1).randint(min_image_size, max_image_size)
            image_height = random.Random(input_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.1, 0.2, 0.3, 0.4]
            ratio = random.Random(input_seed + 3).choice(ratios)
            image = image_create_random_with_two_colors(image_width, image_height, 0, 1, ratio, input_seed + 4)

            layer0_width = random.Random(input_seed + 4).randint(1, image_width)
            layer0_height = random.Random(input_seed + 5).randint(1, image_height)
            if layer0_width == 1 and layer0_height == 1:
                continue

            layer0_x = random.Random(input_seed + 6).randint(0, image_width - layer0_width)
            layer0_y = random.Random(input_seed + 7).randint(0, image_height - layer0_height)
            layer0_rect = Rectangle(
                layer0_x,
                layer0_y,
                layer0_width,
                layer0_height
            )
            # print(f"layer0_rect: {layer0_rect}")
            layer0_image = image_rect(image, layer0_rect, 2)

            mask_image = image_create(image_width, image_height, 0)
            layer0_mask = image_rect(mask_image, layer0_rect, 1)

            layer1_width = random.Random(input_seed + 8).randint(1, image_width)
            layer1_height = random.Random(input_seed + 9).randint(1, image_height)
            layer1_x = random.Random(input_seed + 10).randint(0, image_width - layer1_width)
            layer1_y = random.Random(input_seed + 11).randint(0, image_height - layer1_height)
            layer1_rect = Rectangle(
                layer1_x,
                layer1_y,
                layer1_width,
                layer1_height
            )
            overlap_rect = layer0_rect.intersection(layer1_rect)
            if overlap_rect.is_empty():
                continue
            if overlap_rect == layer0_rect:
                # print(f"overlap the entire layer0, skip")
                continue

            # if the overlap obscures the edge of the rectangle, then it's not possible to repair the obscured area, since the rectangle have no size info.
            corner_topleft = Rectangle(layer0_x, layer0_y, 1, 1)
            corner_topright = Rectangle(layer0_x + layer0_width - 1, layer0_y, 1, 1)
            corner_bottomleft = Rectangle(layer0_x, layer0_y + layer0_height - 1, 1, 1)
            corner_bottomright = Rectangle(layer0_x + layer0_width - 1, layer0_y + layer0_height - 1, 1, 1)

            hidden_topleft = corner_topleft.intersection(layer1_rect).is_not_empty()
            hidden_topright = corner_topright.intersection(layer1_rect).is_not_empty()
            hidden_bottomleft = corner_bottomleft.intersection(layer1_rect).is_not_empty()
            hidden_bottomright = corner_bottomright.intersection(layer1_rect).is_not_empty()

            hidden_top = hidden_topleft and hidden_topright
            hidden_bottom = hidden_bottomleft and hidden_bottomright
            hidden_left = hidden_topleft and hidden_bottomleft
            hidden_right = hidden_topright and hidden_bottomright
            if hidden_top or hidden_bottom or hidden_left or hidden_right:
                # print(f"hidden_top: {hidden_top}, hidden_bottom: {hidden_bottom}, hidden_left: {hidden_left}, hidden_right: {hidden_right}")
                # The obscured area is not repairable.
                continue

            layer1_image = image_rect(layer0_image, layer1_rect, 3)

            input_image = image_replace_colors(layer1_image, color_mapping_input)
            output_image = image_replace_colors(layer0_mask, color_mapping_output)
            break
        if input_image is None or output_image is None:
            raise Exception("Failed to create a pair.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(100):
        if i % 2 == 0:
            task = generate_task_mask_of_primary_rectangle(i)
        else:
            task = generate_task_mask_of_obscured_rectangle(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    if seed % 2 == 0:
        task = generate_task_mask_of_primary_rectangle(seed)
        transformation_id = 'mask_of_primary_rectangle'
    else:
        task = generate_task_mask_of_obscured_rectangle(seed)
        transformation_id = 'mask_of_obscured_rectangle'
        
    # task.show()
    items = generate_dataset_item_list_inner((seed + 1) * 11, task, transformation_id)
    return items


generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=157100911,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
