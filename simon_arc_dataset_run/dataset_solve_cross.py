# Two crossing lines.
#
# IDEA: X marks the spot, where the lines intersect, and a 3x3 box around it.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=67a423a3
#
# IDEA: Parameter for line thickness. Currently lines are always 1 pixel thick.
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
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.image_bresenham_line import *
from simon_arc_lab.image_mask import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_cross'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_cross.jsonl')

def generate_task_two_crossing_lines(seed: int) -> Task:
    """
    Draw 2 crossing lines on an image.
    Highlight the lines that have a particular direction.
    Highlight the intersection of the 2 lines.
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 15

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)
    color_map_input = {}
    for i, color in enumerate(input_colors):
        color_map_input[i] = color

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(output_colors)
    color_map_output = {}
    for i, color in enumerate(output_colors):
        color_map_output[i] = color

    # Decide what the input should be
    available_input_ids = ['mask', 'color', 'intersection']
    input_id = random.Random(seed + 7).choice(available_input_ids)

    # Decide what the output should be
    available_line_ids = ['left_right', 'top_bottom', 'topleft_bottomright', 'topright_bottomleft']
    # pick two unique elements
    line_ids = random.Random(seed + 5).sample(available_line_ids, 2)
    available_output_ids = ['intersection'] + line_ids
    output_id = random.Random(seed + 6).choice(available_output_ids)
    if input_id == 'intersection':
        output_id = 'bothlines'

    intersection_variant = random.Random(seed + 8).randint(0, 2)
    hide_intersection_point = intersection_variant == 1
    use_different_color_for_intersection_point = intersection_variant == 2

    pretty_line_ids = '_'.join(line_ids)
    pretty_line_ids = pretty_line_ids.replace('left_right', 'lr')
    pretty_line_ids = pretty_line_ids.replace('top_bottom', 'tb')
    pretty_line_ids = pretty_line_ids.replace('topleft_bottomright', 'tlbr')
    pretty_line_ids = pretty_line_ids.replace('topright_bottomleft', 'trbl')
    task.metadata_task_id = f'cross {pretty_line_ids} {input_id}{intersection_variant} {output_id}'

    has_diagonal_lines = 'topleft_bottomright' in line_ids or 'topright_bottomleft' in line_ids
    # print(f"has_diagonal_lines: {has_diagonal_lines}")

    for i in range(count_example+count_test):
        is_example = i < count_example
        is_test = i >= count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)
            max_size = max(width, height)

            # This is where the 2 lines are going to intersect
            x = random.Random(iteration_seed + 7).randint(0, width-1)
            y = random.Random(iteration_seed + 8).randint(0, height-1)

            # It's not possible to determine where 2 lines intersect when it's on a corner. 
            # Skip the corners, when it's a test pair.
            # Allow corners for examples, so there is some ambiguity.
            is_x0 = x == 0
            is_x1 = x == width-1
            is_y0 = y == 0
            is_y1 = y == height-1
            is_x0y0 = is_x0 and is_y0
            is_x1y0 = is_x1 and is_y0
            is_x0y1 = is_x0 and is_y1
            is_x1y1 = is_x1 and is_y1
            position_in_corner = is_x0y0 or is_x1y0 or is_x0y1 or is_x1y1
            if is_test and has_diagonal_lines and position_in_corner:
                # print("Skip corner")
                continue

            accumulated_image_or = image_create(width, height, 0)
            accumulated_image_xor = image_create(width, height, 0)
            accumulated_image_intersection = image_create(width, height, 0)
            accumulated_image_lr = image_create(width, height, 0)
            accumulated_image_tb = image_create(width, height, 0)
            accumulated_image_tlbr = image_create(width, height, 0)
            accumulated_image_trbl = image_create(width, height, 0)

            drawing_image = image_create(width, height, 0)

            # Draw the 2 lines
            for j, line_id in enumerate(line_ids):
                image = image_create(width, height, 0)
                draw_color = j + 1

                if line_id == 'left_right':
                    image = image_bresenham_line(image, 0, y, width-1, y, 1)
                    drawing_image = image_bresenham_line(drawing_image, 0, y, width-1, y, draw_color)
                    accumulated_image_lr = image_mask_or(accumulated_image_lr, image)
                elif line_id == 'top_bottom':
                    image = image_bresenham_line(image, x, 0, x, height-1, 1)
                    drawing_image = image_bresenham_line(drawing_image, x, 0, x, height-1, draw_color)
                    accumulated_image_tb = image_mask_or(accumulated_image_tb, image)
                elif line_id == 'topleft_bottomright':
                    image = image_bresenham_line(image, x - max_size, y - max_size, x + max_size, y + max_size, 1)
                    drawing_image = image_bresenham_line(drawing_image, x - max_size, y - max_size, x + max_size, y + max_size, draw_color)
                    accumulated_image_tlbr = image_mask_or(accumulated_image_tlbr, image)
                elif line_id == 'topright_bottomleft':
                    image = image_bresenham_line(image, x + max_size, y - max_size, x - max_size, y + max_size, 1)
                    drawing_image = image_bresenham_line(drawing_image, x + max_size, y - max_size, x - max_size, y + max_size, draw_color)
                    accumulated_image_trbl = image_mask_or(accumulated_image_trbl, image)

                intersection_mask = image_mask_and(accumulated_image_or, image)
                accumulated_image_or = image_mask_or(accumulated_image_or, image)
                accumulated_image_xor = image_mask_xor(accumulated_image_xor, image)
                accumulated_image_intersection = image_mask_or(accumulated_image_intersection, intersection_mask)

            if hide_intersection_point:
                drawing_image[y, x] = 0
            elif use_different_color_for_intersection_point:
                drawing_image[y, x] = 3

            # Prepare input image
            input_image_raw = None
            if input_id == 'mask':
                input_image_raw = accumulated_image_xor
            elif input_id == 'color':
                input_image_raw = drawing_image
            elif input_id == 'intersection':
                input_image_raw = accumulated_image_intersection
            input_image = image_replace_colors(input_image_raw, color_map_input)

            # We are not interested in an empty image
            histogram_input = Histogram.create_with_image(input_image)
            if histogram_input.number_of_unique_colors() < 2:
                continue

            # Prepare output image
            output_image_raw = None
            if output_id == 'left_right':
                output_image_raw = accumulated_image_lr
            elif output_id == 'top_bottom':
                output_image_raw = accumulated_image_tb
            elif output_id == 'topleft_bottomright':
                output_image_raw = accumulated_image_tlbr
            elif output_id == 'topright_bottomleft':
                output_image_raw = accumulated_image_trbl
            elif output_id == 'intersection':
                output_image_raw = accumulated_image_intersection
            elif output_id == 'bothlines':
                output_image_raw = accumulated_image_or
            output_image = image_replace_colors(output_image_raw, color_map_output)

            # We are not interested in an empty image
            histogram_output = Histogram.create_with_image(output_image)
            if histogram_output.number_of_unique_colors() < 2:
                continue

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
    task = generate_task_two_crossing_lines(seed)
    transformation_id = task.metadata_task_id
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1400023425,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
