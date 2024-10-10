# Fractals transformations.
# - create a fractal from a pattern.
# - create a pattern from a fractal.
# - scale up the fractal by 1-3, scale up the pattern by 1-3.
# - invert the pattern.
# - add padding around the input image.
# 
# IDEA: Use image_mask that is multi colored, in order to solve these tasks:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8f2ea7aa
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=27f8ce4f
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=48f8583b
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ad7e01d0
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=15696249
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3af2c5a8
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c3e719e8
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cce03e0d
#
# IDEA: Instead of drawing with a solid color, then use a pattern to draw with, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2072aba6
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ce22a75a
#
# IDEA: Draw image_pattern of 2 scales, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8719f442
#
# IDEA: Split view, one side is the template image, the other side is the tile image, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b4a43f3b_v2
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=363442ee
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b190f7f5
#
# IDEA: Repair the masked out cells, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9ddd00f0
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f9012d9b
#
# IDEA: Two object areas, template image and tile image, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6ecd11f4
#
# IDEA: Swap the output colors.
#
# IDEA: rotate the pattern_image by 90, 180, 270 degrees.
#
# IDEA: use different colors for the output image, than the input image.
#
# IDEA: use different colors in each pair. Currently uses the same global colors for all pairs.
#
# IDEA: use another tile for the output, than the input pattern.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.image_compress import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_fractal import *
from simon_arc_lab.image_pad import *
from simon_arc_lab.image_scale import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.task_formatter_rle_compact import TaskFormatterRLECompact
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_fractal'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_fractal.jsonl')

def generate_task_pattern_to_fractal(seed: int) -> Task:
    """
    Create a fractal image from an input pattern, by repeating the input pattern the places where the mask is 1.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)

    scale_input = random.Random(seed + 11).randint(1, 5)
    scale_output = random.Random(seed + 12).randint(1, 5)
    is_inverse_mask = random.Random(seed + 14).choice([False, True])
    is_padded = random.Random(seed + 16).choice([False, True])
    empty_color = random.Random(seed + 15).choice([0, 1])

    # count_test = 1
    task = Task()
    task.metadata_task_id = f"pattern_to_fractal in={scale_input} out={scale_output} inv={is_inverse_mask} empty={empty_color} pad={is_padded}"
    min_image_size = 2
    max_image_size = 5
    max_pad_count = 8

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 13).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_padding = colors[2]

    color_mapping = {
        0: color0,
        1: color1,
    }

    color_mapping_swap01 = {
        0: 1,
        1: 0,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        pattern_mask = None
        fractal_mask = None
        for retry_index in range(100):
            input_image_seed = (retry_index * 10000) + (seed * 37) + 101 + i * 1333
            width = random.Random(input_image_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(input_image_seed + 2).randint(min_image_size, max_image_size)
            ratio = 0.5
            pattern_mask = image_create_random_with_two_colors(width, height, 0, 1, ratio, input_image_seed + 3)
            if is_inverse_mask:
                pattern_image = image_replace_colors(pattern_mask, color_mapping_swap01)
            else:
                pattern_image = pattern_mask.copy()

            pattern_mask_compressed = image_compress_xy(pattern_mask)
            if pattern_mask_compressed.shape[0] < 2 or pattern_mask_compressed.shape[1] < 2:
                # if it's all a single row/column, then it's not obvious for a human that it's a fractal.
                # the patterns must be easy to spot for a human.
                continue

            histogram = Histogram.create_with_image(pattern_mask)
            if histogram.number_of_unique_colors() != 2:
                # both value 0 and value 1 must be present in the mask.
                continue

            fractal_mask = image_fractal_from_mask_and_image(pattern_mask, pattern_image, empty_color)
            break
        if pattern_mask is None:
            raise Exception("Failed to find a non-trivial example.")
        if fractal_mask is None:
            raise Exception("Failed to find a non-trivial example.")
        input_image = image_replace_colors(pattern_mask, color_mapping)
        output_image = image_replace_colors(fractal_mask, color_mapping)

        input_image = image_scale_uniform(input_image, 'up', scale_input)[1]
        if is_padded:
            input_image = image_pad_random(input_image, seed=seed + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)

        output_image = image_scale_uniform(output_image, 'up', scale_output)[1]
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_fractal_to_pattern(seed: int) -> Task:
    """
    From a fractal image create the pattern.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)

    scale_input = random.Random(seed + 11).randint(1, 5)
    scale_output = random.Random(seed + 12).randint(1, 5)
    is_inverse_mask = random.Random(seed + 14).choice([False, True])
    is_padded = random.Random(seed + 16).choice([False, True])
    empty_color = random.Random(seed + 15).choice([0, 1])

    # count_test = 1
    task = Task()
    task.metadata_task_id = f"fractal_to_pattern in={scale_input} out={scale_output} inv={is_inverse_mask} empty={empty_color} pad={is_padded}"
    min_image_size = 2
    max_image_size = 5
    max_pad_count = 8

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 13).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_padding = colors[2]

    color_mapping = {
        0: color0,
        1: color1,
    }

    color_mapping_swap01 = {
        0: 1,
        1: 0,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        pattern_mask = None
        fractal_mask = None
        for retry_index in range(100):
            input_image_seed = (retry_index * 10000) + (seed * 37) + 101 + i * 1333
            width = random.Random(input_image_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(input_image_seed + 2).randint(min_image_size, max_image_size)
            ratio = 0.5
            pattern_mask = image_create_random_with_two_colors(width, height, 0, 1, ratio, input_image_seed + 3)
            if is_inverse_mask:
                pattern_image = image_replace_colors(pattern_mask, color_mapping_swap01)
            else:
                pattern_image = pattern_mask.copy()

            pattern_mask_compressed = image_compress_xy(pattern_mask)
            if pattern_mask_compressed.shape[0] < 2 or pattern_mask_compressed.shape[1] < 2:
                # if it's all a single row/column, then it's not obvious for a human that it's a fractal.
                # the patterns must be easy to spot for a human.
                continue

            histogram = Histogram.create_with_image(pattern_mask)
            if histogram.number_of_unique_colors() != 2:
                # both value 0 and value 1 must be present in the mask.
                continue

            fractal_mask = image_fractal_from_mask_and_image(pattern_mask, pattern_image, empty_color)
            break
        if pattern_mask is None:
            raise Exception("Failed to find a non-trivial example.")
        if fractal_mask is None:
            raise Exception("Failed to find a non-trivial example.")
        input_image = image_replace_colors(fractal_mask, color_mapping)
        output_image = image_replace_colors(pattern_mask, color_mapping)

        input_image = image_scale_uniform(input_image, 'up', scale_input)[1]
        if is_padded:
            input_image = image_pad_random(input_image, seed=seed + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)

        output_image = image_scale_uniform(output_image, 'up', scale_output)[1]
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def can_fit_inside_context_length(task: Task) -> bool:
    try:
        task_formatter = TaskFormatterRLECompact(task)
        s = task_formatter.to_string()
        return len(s) < 800
    except Exception as e:
        return False

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 2
    task = None
    for retry_index in range(100):
        iteration_seed = seed + retry_index * 100324
        if j == 0:
            transformation_id = 'pattern_to_fractal'
            task = generate_task_pattern_to_fractal(iteration_seed)
        else:
            transformation_id = 'fractal_to_pattern'
            task = generate_task_fractal_to_pattern(iteration_seed)
        
        if can_fit_inside_context_length(task):
            # Bingo, we found a task that fits inside the context length of the LLM.
            break

        # Task is too long to fit inside the context length of LLM. Try again.

    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=15031103031,
    max_num_samples=1000,
    max_byte_size=1024*1024*150
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
