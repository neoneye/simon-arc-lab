# Repair the masked area.
# - identify where the mask is located.
# - repair the masked area.
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
from simon_arc_lab.image_pattern import image_pattern_lines_slope_advanced
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_mask'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_mask.jsonl')

def generate_task_linepatterns_with_masked_areas(seed: int, transformation_id: str) -> Task:
    """
    Generate line patterns, where the input gets masked, and the output is the identified mask, or repair the mask.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = transformation_id
    min_image_size = 4
    max_image_size = 7

    color_map_eliminate_mask_color = {
        0: 1,
    }

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(input_colors)
    color_map_input = {}
    for i in range(10):
        color_map_input[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    color_map_output = {
        0: output_colors[0],
        1: output_colors[1],
    }

    for i in range(100):
        iteration_seed = seed + i * 1000000
        dx = random.Random(iteration_seed + 1).randint(-2, 2)
        dy = random.Random(iteration_seed + 2).randint(-2, 2)
        if dx != 0 or dy != 0:
            break
    if dx == 0 and dy == 0:
        raise Exception("Failed to find a dx, dy that are non-zero.")
    
    line_colors = []
    color_count = random.Random(seed + 20).randint(2, 4)
    for i in range(color_count):
        line_colors.append(input_colors[i])

    square_size = random.Random(seed + 20).randint(1, 2)

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            offsetx = random.Random(iteration_seed + 3).randint(0, 2)
            offsety = random.Random(iteration_seed + 4).randint(0, 2)

            random_a_image_raw = image_pattern_lines_slope_advanced(width, height, dx, dy, square_size, offsetx, offsety, line_colors)

            random_a_image = image_replace_colors(random_a_image_raw, color_map_eliminate_mask_color)
            histogram_a = Histogram.create_with_image(random_a_image)
            if histogram_a.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            ratios = [0.1, 0.2]
            ratio_b = random.Random(iteration_seed + 7).choice(ratios)
            random_b_image = image_create_random_with_two_colors(width, height, 0, 1, ratio_b, iteration_seed + 8)
            histogram_b = Histogram.create_with_image(random_b_image)
            if histogram_b.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            image_with_masked_areas = random_a_image.copy()
            for y in range(height):
                for x in range(width):
                    if random_b_image[y, x] == 1:
                        image_with_masked_areas[y, x] = 0

            histogram_c = Histogram.create_with_image(image_with_masked_areas)
            if histogram_c.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            input_image = image_replace_colors(image_with_masked_areas, color_map_input)
            if transformation_id == 'identify_the_masked_area':
                output_image = image_replace_colors(random_b_image, color_map_output)
            elif transformation_id == 'repair_the_masked_area':
                output_image = image_replace_colors(random_a_image, color_map_input)
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_linepatterns_with_masked_areas(i, 'identify_the_masked_area')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_ids = ['identify_the_masked_area', 'repair_the_masked_area']
    accumulated_dataset_items = []
    for index, transformation_id in enumerate(transformation_ids):
        iteration_seed = seed * 1000000 + index
        task = generate_task_linepatterns_with_masked_areas(iteration_seed + 1, transformation_id)
        dataset_items = generate_dataset_item_list_inner(iteration_seed + 2, task, transformation_id)
        task.show()
        accumulated_dataset_items.extend(dataset_items)
    return accumulated_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=100033371,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)