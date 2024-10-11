# Identify the bounding boxes of objects in the image.
# - Extract one bounding box from multiple lonely pixels. Where it's filled with 1s.
# - Extract one bounding box from multiple lonely pixels. Where it's hollow with the border set to 1s.
#
# IDEA: find the outer bounding box
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4347f46a
#
# IDEA: preserve the original object inside the bounding box, but fill the bounding box with a different color.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6d75e8bb
#
# IDEA: draw the center line/point of the bounding box.
#
# IDEA: draw the bounding box where the border is 1s and the inside is 2s, and the outside is 0s.
#
# IDEA: extend the bounding box, so it draws the top/bottom/left/right border of the box all the way to the edge of the image.
#
# IDEA: draw a box around the object, where it's a hollow box that surrounds the bounding box.
#
# IDEA: show multiple bounding boxes, from different objects, grouped by color
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=56ff96f3
#
# IDEA: show multiple bounding boxes, from different objects, grouped by connected components
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=60b61512
#
# IDEA: bounding boxes of multiple objects ignoring noisy pixels
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7f4411dc
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
from simon_arc_lab.image_paste import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.task import *
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.image_rect import image_rect, image_rect_hollow
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.find_bounding_box import find_bounding_box_ignoring_color
from simon_arc_lab.generate_random_values import GenerateRandomValues
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_boundingbox'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_boundingbox.jsonl')

def generate_task_boundingbox_of_lonely_pixels(seed: int, transformation_id: str) -> Task:
    """
    Show a few lonely pixels, and identify the bounding box.

    Example of filled bounding box:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6d75e8bb

    Example of hollow bounding box:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4347f46a
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e7639916
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 30

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)

    color_map_input = {}
    for i in range(10):
        color_map_input[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    output_color0 = output_colors[0]
    output_color1 = output_colors[1]
    color_map_output = {
        0: output_color0,
        1: output_color1,
    }

    task.metadata_task_id = f'boundingbox_of_lonely_pixels {transformation_id}'

    border_size = 1

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            number_of_positions = random.Random(iteration_seed + 1).randint(2, 5)

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
            input_mask = background_image.copy()
            for x, y in positions:
                input_image_raw[y, x] = random.Random(iteration_seed + 5 + x + y).randint(1, 9)
                input_mask[y, x] = 1

            bounding_box = find_bounding_box_ignoring_color(input_mask, 0)
            if bounding_box.mass() < 1:
                continue
            if bounding_box.width == width and bounding_box.height == height:
                continue

            if transformation_id == 'filled':
                output_image_raw = image_rect(background_image, bounding_box, 1)
            elif transformation_id == 'hollow':
                if bounding_box.width <= 2 or bounding_box.height <= 2:
                    continue
                output_image_raw = image_rect_hollow(background_image, bounding_box, 1, border_size)
            elif transformation_id == 'filled_inner':
                rect2 = Rectangle(bounding_box.x + border_size, bounding_box.y + border_size, bounding_box.width - border_size * 2, bounding_box.height - border_size * 2)
                if rect2.mass() < 1:
                    continue
                output_image_raw = image_rect(background_image, rect2, 1)
            elif transformation_id == 'hollow_inner':
                rect2 = Rectangle(bounding_box.x + border_size, bounding_box.y + border_size, bounding_box.width - border_size * 2, bounding_box.height - border_size * 2)
                if rect2.mass() < 1:
                    continue
                output_image_raw = image_rect_hollow(background_image, rect2, 1, border_size)
            elif transformation_id == 'filled_outer':
                rect2 = Rectangle(bounding_box.x - border_size, bounding_box.y - border_size, bounding_box.width + border_size * 2, bounding_box.height + border_size * 2)
                if rect2.mass() < 1:
                    continue
                output_image_raw = image_rect(background_image, rect2, 1)
            elif transformation_id == 'hollow_outer':
                rect2 = Rectangle(bounding_box.x - border_size, bounding_box.y - border_size, bounding_box.width + border_size * 2, bounding_box.height + border_size * 2)
                if rect2.mass() < 1:
                    continue
                output_image_raw = image_rect_hollow(background_image, rect2, 1, border_size)
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            # We are not interested in images with zero lonely pixels
            histogram = Histogram.create_with_image(output_image_raw)
            if histogram.number_of_unique_colors() < 2:
                continue

            input_image = image_replace_colors(input_image_raw, color_map_input)
            output_image = image_replace_colors(output_image_raw, color_map_output)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_inner_boundingbox(seed: int, transformation_id: str) -> Task:
    """
    Identify the inner bounding box.

    Example:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d37a1ef5
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    max_image_size = 18

    available_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(available_colors)

    color_map = {}
    for i in range(10):
        color_map[i] = available_colors[i]

    task.metadata_task_id = f'boundingbox_inner {transformation_id}'

    gvr = GenerateRandomValues()
    gvr.append_value(1, 6)
    gvr.append_value(1, 6)
    gvr.append_value(2, 5)
    gvr.append_value(1, 6)
    gvr.append_value(1, 6)

    color_background = 0
    color_outer = 1
    color_inner0 = color_background
    color_inner1 = 2
    color_inner2 = 3
    weight0 = 1
    weight1 = 1
    weight2 = 1
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            xvalues = gvr.find_random_values(iteration_seed + 1, max_image_size)
            outer_left = xvalues[0]
            inner_left = xvalues[1]
            inner_width = xvalues[2]
            inner_right = xvalues[3]
            outer_right = xvalues[4]
            outer_width = inner_left + inner_width + inner_right
            background_width = outer_left + outer_width + outer_right

            yvalues = gvr.find_random_values(iteration_seed + 2, max_image_size)
            outer_top = yvalues[0]
            inner_top = yvalues[1]
            inner_height = yvalues[2]
            inner_bottom = yvalues[3]
            outer_bottom = yvalues[4]
            outer_height = inner_top + inner_height + inner_bottom
            background_height = outer_top + outer_height + outer_bottom

            if inner_width + 2 == outer_width and inner_height + 2 == outer_height:
                continue
            if outer_width + 2 == background_width and outer_height + 2 == background_height:
                continue

            inner_image = image_create_random_with_three_colors(inner_width, inner_height, color_inner0, color_inner1, color_inner2, weight0, weight1, weight2, iteration_seed + 20)
            bounding_box = find_bounding_box_ignoring_color(inner_image, color_inner0)
            if bounding_box.mass() < 1:
                continue
            if bounding_box.width != inner_width or bounding_box.height != inner_height:
                # print("bounding_box.width == inner_width and bounding_box.height == inner_height")
                continue

            background_image = image_create(background_width, background_height, color_background)
            input_image_raw = image_rect_hollow(background_image, Rectangle(outer_left, outer_top, outer_width, outer_height), color_outer, 1)
            input_image_raw = image_paste_at(inner_image, input_image_raw, outer_left + inner_left, outer_top + inner_top)

            if transformation_id == 'simple_fill':
                output_background_color = color_background
                output_outer_color = color_outer
            elif transformation_id == 'swap_background_outer':
                output_background_color = color_outer
                output_outer_color = color_background
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            output_background_image = image_create(background_width, background_height, output_background_color)
            output_image_raw = image_rect(output_background_image, Rectangle(outer_left, outer_top, outer_width, outer_height), output_outer_color)
            output_image_raw = image_paste_at(inner_image, output_image_raw, outer_left + inner_left, outer_top + inner_top)

            input_image = image_replace_colors(input_image_raw, color_map)
            output_image = image_replace_colors(output_image_raw, color_map)

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
    j = seed % 8
    # j = (seed % 2) + 6
    # j = 7
    if j == 0:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled')
    elif j == 1:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow')
    elif j == 2:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled_inner')
    elif j == 3:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow_inner')
    elif j == 4:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled_outer')
    elif j == 5:
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow_outer')
    elif j == 6:
        task = generate_task_inner_boundingbox(seed, 'simple_fill')
    elif j == 7:
        task = generate_task_inner_boundingbox(seed, 'swap_background_outer')
    transformation_id = task.metadata_task_id
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=230100913,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
