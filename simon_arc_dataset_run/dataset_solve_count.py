# Count number of lonely pixels, and draw a pattern N times in the output.
#
# IDEA: Add random padding around the input.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_paste import image_paste_random
from simon_arc_lab.image_create_random_advanced import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_count'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_count.jsonl')

def generate_task_count_pixels_and_repeat_output_pattern(seed: int, transformation_id: str) -> Task:
    """
    The input images shows N lonely pixels.

    The output images shows a pattern that repeats N times.

    Example:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4852f2fa_v2
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 14
    min_pattern_size = 1
    max_pattern_size = 5
    min_count = 1
    max_count = 6

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)

    color_map_input = {}
    for i in range(10):
        color_map_input[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    output_color0 = output_colors[0]
    output_color1 = output_colors[1]
    color_map_output = {
        0: output_color0,
        1: output_color1,
    }

    task.metadata_task_id = f'count_pixels_and_repeat_output_pattern {transformation_id}'

    # The pattern image that is shared among the output images
    pattern_width = random.Random(seed + 1).randint(min_pattern_size, max_pattern_size)
    pattern_height = random.Random(seed + 2).randint(min_pattern_size, max_pattern_size)
    pattern_image = image_create_random_advanced(seed + 6, pattern_width, pattern_width, pattern_height, pattern_height)

    # Ensure the number of positions vary among the examples, so it's not all just the same.
    count_list = None
    for retry_index in range(100):
        candidate_count_list = []
        for i in range(count_example+count_test):
            count = random.Random(seed + retry_index * 102344 + i * 83832 + 2).randint(min_count, max_count)
            candidate_count_list.append(count)

        # count the number of unique values among the examples pairs
        count_list_examples = candidate_count_list[:count_example]
        unique_count = len(set(count_list_examples))
        if unique_count < 3:
            # Failed to find enough unique values.
            continue

        count_list = candidate_count_list
        break

    if count_list is None:
        raise Exception("Failed to find count_list with enough unique values.")

    for i in range(count_example+count_test):
        is_example = i < count_example
        number_of_positions = count_list[i]

        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            positions = []
            for i in range(number_of_positions):
                x = random.Random(iteration_seed + 3 + i * 2).randint(0, width - 1)
                y = random.Random(iteration_seed + 4 + i * 2).randint(0, height - 1)
                xy = (x, y)
                if xy in positions:
                    continue
                positions.append(xy)
            if len(positions) != number_of_positions:
                # One or more coordinate collisions. Skip this.
                continue

            background_image = image_create(width, height, 0)
            input_mask = background_image.copy()
            for x, y in positions:
                input_mask[y, x] = 1

            if transformation_id == 'repeat_x':
                input_image_raw = input_mask
                output_image_raw = np.tile(pattern_image, (1, number_of_positions))
            elif transformation_id == 'repeat_y':
                input_image_raw = input_mask
                output_image_raw = np.tile(pattern_image, (number_of_positions, 1))
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            input_image = image_replace_colors(input_image_raw, color_map_input)
            output_image = image_replace_colors(output_image_raw, color_map_output)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_count_pixels_and_repeat_input_pattern(seed: int, transformation_id: str) -> Task:
    """
    The input images shows N lonely pixels, and a pattern area.

    The output images repeats N times the input pattern.

    Example:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4852f2fa_v2
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 4
    max_image_size = 14
    min_pattern_size = 2
    max_pattern_size = 4
    min_count = 1
    max_count = 5

    color_background = 0
    color_indicator = 1

    # Ensures the pattern doesn't have the same colors as the background or the indicators.
    color_map_pattern = {
        color_background: 2,
        color_indicator: 3,
    }

    # The colors used in the input and output images.
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    task.metadata_task_id = f'count_pixels_and_repeat_input_pattern {transformation_id}'

    # Ensure the number of positions vary among the examples, so it's not all just the same.
    count_list = None
    for retry_index in range(100):
        candidate_count_list = []
        for i in range(count_example+count_test):
            count = random.Random(seed + retry_index * 102344 + i * 83832 + 2).randint(min_count, max_count)
            candidate_count_list.append(count)

        # count the number of unique values among the examples pairs
        count_list_examples = candidate_count_list[:count_example]
        unique_count = len(set(count_list_examples))
        if unique_count < 3:
            # Failed to find enough unique values.
            continue

        count_list = candidate_count_list
        break

    if count_list is None:
        raise Exception("Failed to find count_list with enough unique values.")

    for i in range(count_example+count_test):
        is_example = i < count_example
        number_of_positions = count_list[i]

        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            # Generate a random pattern image that doesn't have the background color nor the indicator color.
            pattern_image = image_create_random_advanced(iteration_seed + 3, min_pattern_size, max_pattern_size, min_pattern_size, max_pattern_size)
            pattern_image = image_replace_colors(pattern_image, color_map_pattern)
            histogram = Histogram.create_with_image(pattern_image)
            if histogram.number_of_unique_colors() < 3:
                # Too few colors in the pattern. Skip this.
                continue

            pattern_height, pattern_width = pattern_image.shape

            # Make sure the image is large enough to contain the pattern
            image_width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            image_width = max(image_width, pattern_width)
            image_height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)
            image_height = max(image_height, pattern_height)

            solid_background_image = image_create(image_width, image_height, color_background)
            background_image = image_paste_random(pattern_image, solid_background_image, iteration_seed + 3)

            positions = []
            for i in range(100):
                if len(positions) == number_of_positions:
                    break
                x = random.Random(iteration_seed + 4 + i * 2).randint(0, image_width - 1)
                y = random.Random(iteration_seed + 5 + i * 2).randint(0, image_height - 1)
                xy = (x, y)
                if xy in positions:
                    continue
                if background_image[y, x] != color_background:
                    continue
                positions.append(xy)
            if len(positions) != number_of_positions:
                # One or more coordinate collisions. Skip this.
                continue

            input_mask = background_image.copy()
            for x, y in positions:
                input_mask[y, x] = color_indicator

            if transformation_id == 'repeat_x':
                input_image_raw = input_mask
                output_image_raw = np.tile(pattern_image, (1, number_of_positions))
            elif transformation_id == 'repeat_y':
                input_image_raw = input_mask
                output_image_raw = np.tile(pattern_image, (number_of_positions, 1))
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            input_image = image_replace_colors(input_image_raw, color_map)
            output_image = image_replace_colors(output_image_raw, color_map)

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 4
    # j = (seed % 2) + 2
    if j == 0:
        transformation_id = 'count_pixels_and_repeat_output_pattern_x'
        task = generate_task_count_pixels_and_repeat_output_pattern(seed, 'repeat_x')
    elif j == 1:
        transformation_id = 'count_pixels_and_repeat_output_pattern_y'
        task = generate_task_count_pixels_and_repeat_output_pattern(seed, 'repeat_y')
    elif j == 2:
        transformation_id = 'count_pixels_and_repeat_input_pattern_x'
        task = generate_task_count_pixels_and_repeat_input_pattern(seed, 'repeat_x')
    elif j == 3:
        transformation_id = 'count_pixels_and_repeat_input_pattern_y'
        task = generate_task_count_pixels_and_repeat_input_pattern(seed, 'repeat_y')
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=19000003,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
