# Flip x/y/a/b.
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
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_trim import outer_bounding_box_after_trim_with_color
from simon_arc_lab.image_pad import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_flip'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_flip.jsonl')

def generate_task(seed: int, transformation_id: str) -> Task:
    """
    Do various flips with random images. Add padding to some images.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 8
    max_pad_count = 5

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    color_padding = random.Random(seed + 6).randint(0, 9)

    is_padded = random.Random(seed + 7).randint(0, 1) == 1

    padding_str = '_padded' if is_padded else ''

    task.metadata_task_id = f'{transformation_id}{padding_str}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101
            random_image = image_create_random_advanced(iteration_seed, min_image_size, max_image_size, min_image_size, max_image_size)
            height, width = random_image.shape

            if is_padded:
                bounding_box_rect = outer_bounding_box_after_trim_with_color(random_image, color_padding)
                if bounding_box_rect.width != width or bounding_box_rect.height != height:
                    # Trimming this image would remove some border pixels, causing an ambiguous output image.
                    # print("Skipping image due to ambiguous trimming.")
                    continue

            if transformation_id == 'flipx':
                transformed_image = image_flipx(random_image)
            elif transformation_id == 'flipy':
                transformed_image = image_flipy(random_image)
            elif transformation_id == 'flipa':
                transformed_image = image_flip_diagonal_a(random_image)
            elif transformation_id == 'flipb':
                transformed_image = image_flip_diagonal_b(random_image)
            else:
                raise ValueError(f"Unknown transformation_id: {transformation_id}")
            
            # if random_image is the same as the transformed_image, then skip
            if np.array_equal(random_image, transformed_image):
                continue

            if is_padded:
                random_image = image_pad_random(random_image, seed=iteration_seed+1, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)

            input_image = image_replace_colors(random_image, color_map)
            output_image = image_replace_colors(transformed_image, color_map)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_ids = ['flipx', 'flipy', 'flipa', 'flipb']
    accumulated_dataset_items = []
    for index, transformation_id in enumerate(transformation_ids):
        iteration_seed = seed + index * 1000
        task = generate_task(iteration_seed, transformation_id)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(iteration_seed + 1, task, transformation_id)
        accumulated_dataset_items.extend(dataset_items)
    return accumulated_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=190035117,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
