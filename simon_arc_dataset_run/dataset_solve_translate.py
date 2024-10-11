# Translate the image by -2..+2 pixel, up/down/left/right.
#
# IDEA: Move over the edge without wrapping around.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=32e9702f
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
BENCHMARK_DATASET_NAME = 'solve_translate'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_translate.jsonl')

def generate_task(seed: int, dx: int, dy: int, percent_noise: float, transformation_id: str) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 1
    max_image_size = 12
    task.metadata_task_id = transformation_id

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1000 + i, min_image_size, max_image_size, min_image_size, max_image_size)

        transformed_image = image_translate_wrap(input_image, dx, dy)

        height, width = transformed_image.shape
        noise_image = image_create_random_advanced(seed + 1001 + i, width, width, height, height)
        mask = image_create_random_with_two_colors(width, height, 0, 1, percent_noise, seed + 1050 + i)

        output_image = image_mix(mask, transformed_image, noise_image)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    # builder.append_arcagi1_json()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    random.seed(seed)

    seed_task = seed

    max_translate = 3

    # Populate a config_list like this 
    # [(-1, -1, 'translate_xminus1yminus1'), (0, -1, 'translate_yminus1'), (1, -1, 'translate_xplus1yminus1'), 
    #  (-1, 0, 'translate_xminus1'), (1, 0, 'translate_xplus1'), (-1, 1, 'translate_xminus1yplus1'), 
    #  (0, 1, 'translate_yplus1'), (1, 1, 'translate_xplus1yplus1')]
    config_list = []
    for dy in range(-max_translate, max_translate+1):
        for dx in range(-max_translate, max_translate+1):
            if dx == 0 and dy == 0:
                continue
            name_x = 'zero'
            if dx > 0:
                name_x = f'xplus{dx}'
            else:
                name_x = f'xminus{-dx}'
            name_y = 'zero'
            if dy > 0:
                name_y = f'yplus{dy}'
            else:
                name_y = f'yminus{-dy}'
            if dx == 0:
                transformation_id = f'translate_{name_y}'
            elif dy == 0:
                transformation_id = f'translate_{name_x}'
            else:
                transformation_id = f'translate_{name_x}{name_y}'
            config_list.append((dx, dy, transformation_id))

#    print(f'config_list: {config_list}')

    # shuffle the parameters
    random.shuffle(config_list)

    # truncate the parameters to a few
    truncate_length = random.randint(2, 5)
    config_list_truncated = config_list[:truncate_length]

    # print(f'truncate_length: {truncate_length} config_list_truncated: {config_list_truncated}')

    all_dataset_items = []
    for config in config_list_truncated:
        dx, dy, transformation_id = config
        percent_noise = 0.0
        task = generate_task(seed_task, dx, dy, percent_noise, transformation_id)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=62100005,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
