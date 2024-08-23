# Where is a pixel located inside an object?
# - If it's the top/bottom/left/right -facing of the object, then set the mask to 1, otherwise 0.
#
# IDEA: Also exercise ImageShape3x3Center.TOP_LEFT, ImageShape3x3Center.TOP_RIGHT, ImageShape3x3Center.BOTTOM_LEFT, ImageShape3x3Center.BOTTOM_RIGHT
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
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_shape3x3_center import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_edge'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_edge.jsonl')

def generate_task_edge(seed: int, edge_name: str) -> Task:
    """
    Identify the mask of the pixels that are on the edge of an object.
    """

    if edge_name == 'top':
        find_bitmask = ImageShape3x3Center.TOP
    elif edge_name == 'bottom':
        find_bitmask = ImageShape3x3Center.BOTTOM
    elif edge_name == 'left':
        find_bitmask = ImageShape3x3Center.LEFT
    elif edge_name == 'right':
        find_bitmask = ImageShape3x3Center.RIGHT
    elif edge_name == 'top_left':
        find_bitmask = ImageShape3x3Center.TOP_LEFT
    elif edge_name == 'top_right':
        find_bitmask = ImageShape3x3Center.TOP_RIGHT
    elif edge_name == 'bottom_left':
        find_bitmask = ImageShape3x3Center.BOTTOM_LEFT
    elif edge_name == 'bottom_right':
        find_bitmask = ImageShape3x3Center.BOTTOM_RIGHT
    else:
        raise Exception(f"Unknown edge_name: {edge_name}")

    count_example = random.Random(seed + 9).randint(4, 5)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'edge_{edge_name}'
    min_image_size = 3
    max_image_size = 10

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_map_output = {
        0: color0,
        1: color1,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101
            random_image = image_create_random_advanced(iteration_seed, min_image_size, max_image_size, min_image_size, max_image_size)
            height, width = random_image.shape

            bitmask_image = ImageShape3x3Center.apply(random_image)

            # Positions where the bitmask has bit[N] == 1
            mask = np.zeros_like(random_image)
            for y in range(height):
                for x in range(width):
                    if bitmask_image[y, x] & find_bitmask > 0:
                        mask[y, x] = 1

            # We are not interested in images with nothing going on.
            histogram = Histogram.create_with_image(mask)
            if histogram.number_of_unique_colors() < 2:
                continue

            output_image = image_replace_colors(mask, color_map_output)
            input_image = random_image
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_edge(i, 'top')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    name_list = [
        # 'top',
        # 'bottom', 
        # 'left',
        # 'right',
        'top_left',
        'top_right',
        'bottom_left',
        'bottom_right',
    ]
    accumulated_dataset_items = []
    for index, name in enumerate(name_list):
        iteration_seed = seed + 1000000 * index
        task = generate_task_edge(iteration_seed + 1, name)
        # task.show()
        transformation_id = f'edge_{name}'
        dataset_items = generate_dataset_item_list_inner(iteration_seed + 2, task, transformation_id)
        accumulated_dataset_items.extend(dataset_items)

    return accumulated_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=140999999,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
