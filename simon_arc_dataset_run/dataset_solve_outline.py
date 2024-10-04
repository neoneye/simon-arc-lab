# Edge detection, identify the outline of objects.
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
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_outline import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_outline'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_outline.jsonl')

def generate_task_outline_all8(seed: int) -> Task:
    """
    Create a compressed image from an input image, by removing duplicate rows and columns.
    """
    count_example = random.Random(seed + 1).randint(3, 5)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'outline_all8'
    min_image_size = 3
    max_image_size = 10

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
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
            input_image_seed = (retry_index * 10000) + (seed * 37) + 101 + i
            input_image = image_create_random_advanced(input_image_seed, min_image_size, max_image_size, min_image_size, max_image_size)
            outline_mask = image_outline_all8(input_image)

            count_ones = np.count_nonzero(outline_mask == 1)
            count_zeros = np.count_nonzero(outline_mask == 0)
            if count_ones == 0 or count_zeros == 0:
                # there is nothing remaining after the edge detection, skip this image.
                continue

            output_image = image_replace_colors(outline_mask, color_mapping)
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_id = 'outline_all8'
    task = generate_task_outline_all8(seed)
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=817300713,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
