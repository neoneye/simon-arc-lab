# Fractals transformations.
# - create a fractal from a pattern.
# - create a pattern from a fractal.
#
# IDEA: scale up the input image.
#
# IDEA: add padding around the input image.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_fractal import *
from simon_arc_lab.image_compress import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_fractal'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_fractal.jsonl')

def generate_task_pattern_to_fractal(seed: int) -> Task:
    """
    Create a fractal image from an input pattern, by repeating the input pattern the places where the mask is 1.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'pattern_to_fractal'
    min_image_size = 2
    max_image_size = 3

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 11).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_mapping = {
        0: color0,
        1: color1,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_image_seed = (retry_index * 10000) + (seed * 37) + 101 + i * 1333
            width = random.Random(input_image_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(input_image_seed + 2).randint(min_image_size, max_image_size)
            ratio = 0.5
            pattern_mask = image_create_random_with_two_colors(width, height, 0, 1, ratio, input_image_seed + 3)

            pattern_mask_compressed = image_compress_xy(pattern_mask)
            if pattern_mask_compressed.shape[0] < 2 or pattern_mask_compressed.shape[1] < 2:
                # if it's all a single row/column, then it's not obvious for a human that it's a fractal.
                # the patterns must be easy to spot for a human.
                continue

            histogram = Histogram.create_with_image(pattern_mask)
            if histogram.number_of_unique_colors() != 2:
                # both value 0 and value 1 must be present in the mask.
                continue

            fractal_mask = image_fractal_from_mask(pattern_mask)

            input_image = image_replace_colors(pattern_mask, color_mapping)
            output_image = image_replace_colors(fractal_mask, color_mapping)
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_fractal_to_pattern(seed: int) -> Task:
    """
    From a fractal image create the pattern.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'fractal_to_pattern'
    min_image_size = 2
    max_image_size = 3

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 11).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_mapping = {
        0: color0,
        1: color1,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_image_seed = (retry_index * 10000) + (seed * 37) + 101 + i * 1333
            width = random.Random(input_image_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(input_image_seed + 2).randint(min_image_size, max_image_size)
            ratio = 0.5
            pattern_mask = image_create_random_with_two_colors(width, height, 0, 1, ratio, input_image_seed + 3)

            pattern_mask_compressed = image_compress_xy(pattern_mask)
            if pattern_mask_compressed.shape[0] < 2 or pattern_mask_compressed.shape[1] < 2:
                # if it's all a single row/column, then it's not obvious for a human that it's a fractal.
                # the patterns must be easy to spot for a human.
                continue

            histogram = Histogram.create_with_image(pattern_mask)
            if histogram.number_of_unique_colors() != 2:
                # both value 0 and value 1 must be present in the mask.
                continue

            fractal_mask = image_fractal_from_mask(pattern_mask)

            input_image = image_replace_colors(fractal_mask, color_mapping)
            output_image = image_replace_colors(pattern_mask, color_mapping)
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        if i % 2 == 0:
            task = generate_task_pattern_to_fractal(i)
        else:
            task = generate_task_fractal_to_pattern(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    if seed % 2 == 0:
        transformation_id = 'pattern_to_fractal'
        task = generate_task_pattern_to_fractal(seed)
    else:
        transformation_id = 'fractal_to_pattern'
        task = generate_task_fractal_to_pattern(seed)
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=7701103031,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
