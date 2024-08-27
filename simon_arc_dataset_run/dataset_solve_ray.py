# Send out rays from objects.
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
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_trim import outer_bounding_box_after_trim_with_color
from simon_arc_lab.image_bresenham_line import image_bresenham_line
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_ray'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_ray.jsonl')

def generate_task_emit_rays_from_lonely_pixels(seed: int) -> Task:
    """
    Show a few lonely pixels, and shoot out rays in many directions.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 5
    max_image_size = 20

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

    variant_index = seed % 20

    # use 4 bits enabling/disabling the following:
    variant_bits = 0
    if variant_index <= 14:
        variant_bits = variant_index + 1 # values in the range 1-15
    draw_from_topleft_to_bottomright = variant_bits & 1 > 0
    draw_from_topright_to_bottomleft = variant_bits & 2 > 0
    draw_from_left_to_right = variant_bits & 4 > 0
    draw_from_top_to_bottom = variant_bits & 8 > 0

    # use the remaining values for various boxes
    draw_box3x3_hollow  = variant_index == 15
    draw_box3x3_filled  = variant_index == 16
    draw_box5x5_hollow1 = variant_index == 17
    draw_box5x5_hollow2 = variant_index == 18
    draw_box5x5_filled  = variant_index == 19

    task.metadata_task_id = f'emit_rays_from_lonely_pixels_variant{variant_index}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            number_of_positions = random.Random(iteration_seed + 1).randint(1, 3)

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
            accumulated_mask = background_image.copy()
            size = max(width, height)
            for x, y in positions:
                input_image_raw[y, x] = random.Random(iteration_seed + 5 + x + y).randint(1, 9)
                if draw_box3x3_hollow:
                    accumulated_mask = image_rect_hollow(accumulated_mask, Rectangle(x - 1, y - 1, 3, 3), 1, 1)
                if draw_box3x3_filled:
                    accumulated_mask = image_rect(accumulated_mask, Rectangle(x - 1, y - 1, 3, 3), 1)
                if draw_box5x5_hollow1:
                    accumulated_mask = image_rect_hollow(accumulated_mask, Rectangle(x - 2, y - 2, 5, 5), 1, 1)
                if draw_box5x5_hollow2:
                    accumulated_mask = image_rect_hollow(accumulated_mask, Rectangle(x - 2, y - 2, 5, 5), 1, 2)
                if draw_box5x5_filled:
                    accumulated_mask = image_rect(accumulated_mask, Rectangle(x - 2, y - 2, 5, 5), 1)
                if draw_from_topleft_to_bottomright:
                    accumulated_mask = image_bresenham_line(accumulated_mask, x - size, y - size, x + size, y + size, 1)
                if draw_from_topright_to_bottomleft:
                    accumulated_mask = image_bresenham_line(accumulated_mask, x + size, y - size, x - size, y + size, 1)
                if draw_from_left_to_right:
                    accumulated_mask = image_bresenham_line(accumulated_mask, 0, y, width - 1, y, 1)
                if draw_from_top_to_bottom:
                    accumulated_mask = image_bresenham_line(accumulated_mask, x, 0, x, height - 1, 1)

            input_image = image_replace_colors(input_image_raw, color_map_input)
            output_image = image_replace_colors(accumulated_mask, color_map_output)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_emit_rays_from_lonely_pixels(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_id = 'emit_rays_from_lonely_pixels'
    task = generate_task_emit_rays_from_lonely_pixels(seed)
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=120055117,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
