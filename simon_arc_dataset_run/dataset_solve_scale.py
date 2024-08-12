# Scale the image up/down by x/y factor.
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
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_scale import image_scale
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_scale'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_scale.jsonl')

def generate_task(seed: int, x_up_down, x_scale, y_up_down, y_scale) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 3
    max_width = 10
    min_height = 3
    max_height = 10

    for i in range(count_example+count_test):
        is_example = i < count_example
        unscaled_image = image_create_random_advanced(seed + 1000 + i, min_width, max_width, min_height, max_height)

        input_image, output_image = image_scale(unscaled_image, x_up_down, x_scale, y_up_down, y_scale)

        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(3):
        seed = i
        x_scale = random.Random(seed + 3).randint(1, 4)
        y_scale = random.Random(seed + 4).randint(1, 4)
        if x_scale == 1 and y_scale == 1:
            x_scale = 2
            y_scale = 2

        up_down = ['up', 'down']
        x_up_down = random.Random(seed + 5).choice(up_down)
        y_up_down = random.Random(seed + 6).choice(up_down)
        task = generate_task(i, x_up_down, x_scale, y_up_down, y_scale)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_height()
    # builder.append_pixels()
    builder.append_image()
    return builder.dataset_items()

def format_scalexy_identifier(x_up_down: str, x_scale: int, y_up_down: str, y_scale: int) -> str:
    """
    the x_up_down and y_up_down are either 'up' or 'down'
    the x_scale and y_scale are positive integers in the range 1 to max_scale_factor
    """
    if x_scale == 1:
        x_suffix = ''
    else:
        x_suffix = str(x_up_down)
    if y_scale == 1:
        y_suffix = ''
    else:
        y_suffix = str(y_up_down)
    return f'x{x_scale}{x_suffix}_y{y_scale}{y_suffix}'

def generate_dataset_item_list(seed: int) -> list[dict]:
    random.seed(seed)

    seed_task = seed

    max_scale_factor = 3
    up_down = ['up', 'down']
    config_list = []
    for x_up_down in up_down:
        for y_up_down in up_down:
            for y_scale in range(1, max_scale_factor + 1):
                for x_scale in range(1, max_scale_factor + 1):
                    if x_scale == 1 and y_scale == 1:
                        continue
                    transformation_id = format_scalexy_identifier(x_up_down, x_scale, y_up_down, y_scale)
                    config_list.append((x_up_down, x_scale, y_up_down, y_scale, transformation_id))

    # shuffle the parameters
    random.Random(seed + 1).shuffle(config_list)

    # truncate the parameters to a few
    truncate_length = random.randint(2, 5)
    config_list_truncated = config_list[:truncate_length]

    all_dataset_items = []
    for config_list in config_list_truncated:
        x_up_down, x_scale, y_up_down, y_scale, transformation_id = config_list
        task = generate_task(seed_task, x_up_down, x_scale, y_up_down, y_scale)
        # print(transformation_id)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=21000019,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
