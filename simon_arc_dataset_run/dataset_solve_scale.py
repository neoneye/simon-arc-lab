# Scale the image up/down by x/y factor.
# - Images that are scaled down, get some noise added to the input, so the model learns to deal with the noise.
# - Images that are scaled up, does not get any noise added to them, otherwise it wouldn't be possible to up scale them.
#
# Model has been trained with max_image_size: 1-20. Model needs to be train with max_image_size greater than 20.
# Model has been trained with max_scale_factor: 1-7. Model needs to be train with max_scale_factor greater than 7.
#
# IDEA: padding around the input images, that have to be removed by the model.
# IDEA: scale factor that corresponds to the number of unique colors.
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

def generate_task(seed: int, x_up_down, x_scale, y_up_down, y_scale) -> Task:
    """
    Create a task, where the job is to scale the image up or down.
    And in some cases compensate for noise.
    
    Example of task with noise and scale down:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5614dbcf

    Example of task with scale up:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9172f3a0
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c59eb873
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 1
    max_image_size = 7

    can_add_noise = False
    if x_up_down == 'down' and y_up_down == 'down':
        if x_scale > 1 and y_scale > 1:
            can_add_noise = True

    # don't always add noise
    if can_add_noise:
        percent = random.Random(seed + 3).randint(0, 100)
        if percent > 90:
            # print(f"suppressing noise. Perfect down scaling")
            can_add_noise = False

    noise_color = None
    if can_add_noise:
        # 0-9:  half of the time the same color is used
        # None: half of the time, it's a random color
        noise_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None, None, None, None, None]
        noise_color = random.Random(seed + 4).choice(noise_colors)

    if can_add_noise:
        if noise_color is None:
            noise_str = f'_noiserandom'
        else:
            noise_str = f'_noise{noise_color}'
    else:
        noise_str = ''

    scale_id = format_scalexy_identifier(x_up_down, x_scale, y_up_down, y_scale)
    task.metadata_task_id = f'scale_{scale_id}{noise_str}'

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

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_image_randomized()
    builder.append_image_rawpixel_output()
    return builder.dataset_items()

class DatasetSolveScale(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        max_scale_factor = 3
        up_down = ['up', 'down']
        config_list = []
        for x_up_down in up_down:
            for y_up_down in up_down:
                for y_scale in range(1, max_scale_factor + 1):
                    for x_scale in range(1, max_scale_factor + 1):
                        if x_scale == 1 and y_scale == 1:
                            continue
                        config_list.append((x_up_down, x_scale, y_up_down, y_scale))

        # shuffle the parameters
        random.Random(seed + 1).shuffle(config_list)

        # truncate the parameters to a few
        truncate_length = random.Random(seed + 2).randint(2, 5)
        config_list_truncated = config_list[:truncate_length]

        all_dataset_items = []
        for index, config_list in enumerate(config_list_truncated):
            iteration_seed = seed + index * 1000000
            x_up_down, x_scale, y_up_down, y_scale = config_list
            task = generate_task(iteration_seed, x_up_down, x_scale, y_up_down, y_scale)
            transformation_id = task.metadata_task_id
            if show:
                task.show()
            dataset_items = generate_dataset_item_list_inner(iteration_seed + 1, task, transformation_id)
            all_dataset_items.extend(dataset_items)

        return all_dataset_items

if __name__ == "__main__":
    generator = DatasetSolveScale()
    generator.generate(
        seed=82400019,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
