# Boolean operations between 2 images: SAME, XOR, AND, OR.
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
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_bool'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_bool.jsonl')

def image_mask_same(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] == image1[y, x]:
                mask[y, x] = 1
    return mask

def image_mask_and(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] == 1 and image1[y, x] == 1:
                mask[y, x] = 1
    return mask

def image_mask_or(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] == 1 or image1[y, x] == 1:
                mask[y, x] = 1
    return mask

def image_mask_xor(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] != image1[y, x]:
                mask[y, x] = 1
    return mask

def generate_task_bool_transformation(seed: int, transformation_id: str) -> Task:
    """
    Stack two images together and apply a boolean transformation, such as AND, OR, XOR, SAME.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = transformation_id
    min_image_size = 3
    max_image_size = 5

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)

    input_color0 = input_colors[0]
    input_color1 = input_colors[1]
    input_color2 = input_colors[2]
    wall_color = 2
    color_map_input = {
        0: input_color0,
        1: input_color1,
        wall_color: input_color2,
    }

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(output_colors)
    output_color0 = output_colors[0]
    output_color1 = output_colors[1]
    color_map_output = {
        0: output_color0,
        1: output_color1,
    }
    
    wall_size = 1

    is_hstack = random.Random(seed + 5).randint(0, 1) == 0
    has_wall = random.Random(seed + 6).randint(0, 1) == 0

    stack_name = 'hstack' if is_hstack else 'vstack'
    wall_name = '_wall' if has_wall else ''
    task.metadata_task_id = f'{transformation_id}_{stack_name}{wall_name}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
            ratio0 = random.Random(iteration_seed + 5).choice(ratios)
            random_image0 = image_create_random_with_two_colors(width, height, 0, 1, ratio0, iteration_seed + 6)
            histogram_random_image0 = Histogram.create_with_image(random_image0)
            if histogram_random_image0.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            ratio1 = random.Random(iteration_seed + 7).choice(ratios)
            random_image1 = image_create_random_with_two_colors(width, height, 0, 1, ratio1, iteration_seed + 8)
            histogram_random_image1 = Histogram.create_with_image(random_image1)
            if histogram_random_image1.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            # Identify where the pixels are the same
            mask = None
            if transformation_id == 'same':
                mask = image_mask_same(random_image0, random_image1)
            elif transformation_id == 'and':
                mask = image_mask_and(random_image0, random_image1)
            elif transformation_id == 'or':
                mask = image_mask_or(random_image0, random_image1)
            elif transformation_id == 'xor':
                mask = image_mask_xor(random_image0, random_image1)
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")
            
            # We are not interested in empty images
            histogram_mask = Histogram.create_with_image(mask)
            if histogram_mask.number_of_unique_colors() < 2:
                continue

            hstack_wall_image = image_create(wall_size, height, wall_color)
            vstack_wall_image = image_create(width, wall_size, wall_color)

            if is_hstack:
                if has_wall:
                    stacked_image = np.hstack([random_image0, hstack_wall_image, random_image1])
                else:
                    stacked_image = np.hstack([random_image0, random_image1])
            else:
                if has_wall:
                    stacked_image = np.vstack([random_image0, vstack_wall_image, random_image1])
                else:
                    stacked_image = np.vstack([random_image0, random_image1])

            input_image = image_replace_colors(stacked_image, color_map_input)
            output_image = image_replace_colors(mask, color_map_output)
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_bool_transformation(i, 'same')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_ids = [
        'same',
        'and',
        'or',
        'xor',
    ]
    accumulated_dataset_items = []
    for index, transformation_id in enumerate(transformation_ids):
        task = generate_task_bool_transformation(seed + index * 100338383, transformation_id)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        accumulated_dataset_items.extend(dataset_items)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=190000771,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
