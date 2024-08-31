# Identify the bounding boxes of objects in the image.
# - Extract one bounding box from multiple lonely pixels. Where it's filled with 1s.
# - Extract one bounding box from multiple lonely pixels. Where it's hollow with the border set to 1s.
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
from simon_arc_lab.task import *
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.image_rect import image_rect, image_rect_hollow
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_trim import outer_bounding_box_after_trim_with_color
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

            bounding_box = outer_bounding_box_after_trim_with_color(input_mask, 0)
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

def demo_generate_task():
    for i in range(5):
        task = generate_task_boundingbox_of_lonely_pixels(i, 'filled')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 6
    if j == 0:
        transformation_id = 'boundingbox_of_lonely_pixels_filled'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled')
    elif j == 1:
        transformation_id = 'boundingbox_of_lonely_pixels_hollow'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow')
    elif j == 2:
        transformation_id = 'boundingbox_of_lonely_pixels_filled_inner'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled_inner')
    elif j == 3:
        transformation_id = 'boundingbox_of_lonely_pixels_hollow_inner'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow_inner')
    elif j == 4:
        transformation_id = 'boundingbox_of_lonely_pixels_filled_outer'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'filled_outer')
    elif j == 5:
        transformation_id = 'boundingbox_of_lonely_pixels_hollow_outer'
        task = generate_task_boundingbox_of_lonely_pixels(seed, 'hollow_outer')
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=210000913,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
