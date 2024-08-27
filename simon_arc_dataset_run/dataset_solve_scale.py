# Scale the image up/down by x/y factor.
#
# Model has been trained with max_image_size: 1-20. Model needs to be train with max_image_size greater than 20.
# Model has been trained with max_scale_factor: 1-7. Model needs to be train with max_scale_factor greater than 7.
#
# IDEA: add noise to the image being down scaled.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5614dbcf
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

def compute_max_image_size(max_image_size: int, scale_factor: int) -> int:
    computed_max_image_size = max_image_size
    if scale_factor >= 2:
        computed_max_image_size = max_image_size // scale_factor
        if computed_max_image_size < 1:
            computed_max_image_size = 1

    # print(f"scale_factor: {scale_factor} computed_max_image_size {computed_max_image_size}")
    return computed_max_image_size

def scale_up_and_add_noise(unscaled_image: np.array, seed: int, scale_x: int, scale_y: int, noise_color: Optional[int]) -> np.array:
    height, width = unscaled_image.shape
    result_image = np.zeros((height * scale_y, width * scale_x), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            value = unscaled_image[y, x]
            for y_offset in range(scale_y):
                for x_offset in range(scale_x):
                    result_image[y*scale_y + y_offset, x*scale_x + x_offset] = value
    positions = []
    for y_offset in range(scale_y):
        for x_offset in range(scale_x):
            positions.append((x_offset, y_offset))

    max_random_positions = (scale_x * scale_y) // 2 - 1
    if max_random_positions < 1:
        max_random_positions = 1
    for y in range(height):
        for x in range(width):
            positions_copy = positions.copy()
            random.Random(seed + y * 100 + x).shuffle(positions_copy)
            count_positions = random.Random(seed + y * 100 + x + 1).randint(0, max_random_positions)
            for i in range(count_positions):
                x_offset, y_offset = positions_copy[i]
                if noise_color is None:
                    color = random.Random(seed + y * 10000 + x * 3939 + 23838 + i * 38382).randint(0, 9)
                else:
                    color = noise_color
                result_image[y*scale_y + y_offset, x*scale_x + x_offset] = color

    return result_image

def generate_task(seed: int, x_up_down, x_scale, y_up_down, y_scale) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 1
    max_image_size = 12

    can_add_noise = False
    if x_up_down == 'down' and y_up_down == 'down':
        if x_scale > 1 and y_scale > 1:
            can_add_noise = True

    # don't always add noise
    if can_add_noise:
        percent = random.Random(seed + 3).randint(0, 100)
        if percent > 70:
            # print(f"suppressing noise. Perfect down scaling")
            can_add_noise = False

    noise_color = None
    if can_add_noise:
        # half of the time the same color is used
        # half of the time, it's a random color
        noise_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None, None, None, None, None]
        noise_color = random.Random(seed + 4).choice(noise_colors)

    if can_add_noise:
        if noise_color is None:
            noise_str = f'_noiserandom'
        else:
            noise_str = f'_noise{noise_color}'
    else:
        noise_str = ''

    task.metadata_task_id = f'scale_x{x_up_down}{x_scale}_y{y_up_down}{y_scale}{noise_str}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        computed_x_max_image_size = compute_max_image_size(max_image_size, x_scale)
        computed_y_max_image_size = compute_max_image_size(max_image_size, y_scale)
        unscaled_image = image_create_random_advanced(seed + 1000 + i, min_image_size, computed_x_max_image_size, min_image_size, computed_y_max_image_size)

        input_image, output_image = image_scale(unscaled_image, x_up_down, x_scale, y_up_down, y_scale)

        if can_add_noise:
            input_image = scale_up_and_add_noise(unscaled_image, seed + i * 13392 + 1000, x_scale, y_scale, noise_color)

        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(3):
        seed = i
        x_scale = random.Random(seed + 3).randint(1, 6)
        y_scale = random.Random(seed + 4).randint(1, 6)
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
    max_scale_factor = 7
    up_down = ['up', 'down']
    up_down = ['down']
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
    truncate_length = random.Random(seed + 2).randint(2, 5)
    config_list_truncated = config_list[:truncate_length]

    all_dataset_items = []
    for index, config_list in enumerate(config_list_truncated):
        iteration_seed = seed + index * 1000000
        x_up_down, x_scale, y_up_down, y_scale, transformation_id = config_list
        task = generate_task(iteration_seed, x_up_down, x_scale, y_up_down, y_scale)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(iteration_seed + 1, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=71000019,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
