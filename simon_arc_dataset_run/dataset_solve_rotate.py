# Rotate the image by 90 degrees, clockwise, counterclockwise, 180 degrees.
# Does flipx, flipy, flip_diagonal_a, flip_diagonal_b.
#
# IDEA: Add some noise to the example pairs, but not to the test pairs.
# This way the model have to recognize the transformation despite the noise.
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
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_rotate'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_rotate.jsonl')

def generate_task(seed: int, transformation_id: str, percent_noise: float) -> Task:
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_size = 1
    max_size = 13
    task.metadata_task_id = transformation_id

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1002 + i, min_size, max_size, min_size, max_size)

        transformed_image = None
        if transformation_id == 'rotate_cw':
            transformed_image = image_rotate_cw(input_image)
        elif transformation_id == 'rotate_ccw':
            transformed_image = image_rotate_ccw(input_image)
        elif transformation_id == 'rotate_180':
            transformed_image = image_rotate_180(input_image)
        elif transformation_id == 'flipa':
            transformed_image = image_flip_diagonal_a(input_image)
        elif transformation_id == 'flipb':
            transformed_image = image_flip_diagonal_b(input_image)
        elif transformation_id == 'flipx':
            transformed_image = image_flipx(input_image)
        elif transformation_id == 'flipy':
            transformed_image = image_flipy(input_image)
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")

        height, width = transformed_image.shape
        noise_image = image_create_random_advanced(seed + 1001 + i, width, width, height, height)
        mask = image_create_random_with_two_colors(width, height, 0, 1, percent_noise, seed + 1050 + i)

        output_image = image_mix(mask, transformed_image, noise_image)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    random.seed(seed)

    seed_task = seed

    transformation_ids = [
        'rotate_cw',
        'rotate_ccw',
        'rotate_180',
        'flipa',
        'flipb',
        'flipx',
        'flipy',
    ]

    all_dataset_items = []
    for transformation_id in transformation_ids:
        percent_noise = 0.0
        task = generate_task(seed_task, transformation_id, percent_noise)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=139000235,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
