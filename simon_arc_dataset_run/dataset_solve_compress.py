# Compression transformations.
# - remove duplicate rows/columns.
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
from simon_arc_lab.image_compress import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_compress'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_compress.jsonl')

def generate_task_compress_xy(seed: int) -> Task:
    """
    Create a compressed image from an input image, by removing duplicate rows and columns.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'compress_xy'
    min_width = 2
    max_width = 4
    min_height = 2
    max_height = 4

    count_compress_x = 0
    count_compress_y = 0
    count_one_pixel_result = 0
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_image = image_create_random_advanced((retry_index * 10000) + (seed * 37) + 101 + i, min_width, max_width, min_height, max_height)
            image_xy = image_compress_xy(input_image)
            if np.array_equal(input_image, image_xy):
                # The image is already compressed in both directions. Try again.
                continue
            image_x = image_compress_x(input_image)
            image_y = image_compress_y(input_image)
            same_x = np.array_equal(input_image, image_x)
            same_y = np.array_equal(input_image, image_y)
            if same_x == False:
                count_compress_x += 1
            if same_y == False:
                count_compress_y += 1

            if image_xy.shape[0] == 1 and image_xy.shape[1] == 1:
                count_one_pixel_result += 1
                if count_one_pixel_result > 1:
                    # Avoid having too many examples with a single pixel output. Limit to 1 output image.
                    continue

            if i + 1 == count_example:
                if count_compress_x == 0 or count_compress_y == 0:
                    # At least one example of each compression is needed.
                    continue
            output_image = image_xy
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_compress_xy(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_id = 'compressxy'
    task = generate_task_compress_xy(seed)
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=303600313,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
