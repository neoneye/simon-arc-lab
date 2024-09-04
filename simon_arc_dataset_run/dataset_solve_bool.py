# Boolean operations between 2 images: AND, OR, XOR.
#
# IDEA: use same background color when the input_a and input_b are using different colors.
# IDEA: use different colors for each pair, and pass on the input colors to the output.
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

def generate_task_bool_transformation(seed: int, transformation_id: str) -> Task:
    """
    Stack two images together and apply a boolean transformation, such as AND, OR, XOR.

    Example of tasks with 'AND' transformation:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=0520fde7
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1b2d62fb
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6430c8c4
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=94f9d214
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f2829549
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=fafffa47
    
    Example of tasks with 'OR' transformation:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=dae9d2b5
    
    Example of tasks with 'XOR' transformation:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3428a4f5
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=99b1bc43

    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = transformation_id
    min_image_size = 2
    max_image_size = 7

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)

    wall_color = input_colors[9]

    color_map_input_a = {
        0: input_colors[0],
        1: input_colors[1],
    }
    color_map_input_b = {
        0: input_colors[2],
        1: input_colors[3],
    }

    is_same_palette_for_inputs = random.Random(seed + 4).randint(0, 1) == 0
    if is_same_palette_for_inputs:
        color_map_input_b = color_map_input_a

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    output_color0 = output_colors[0]
    output_color1 = output_colors[1]
    color_map_output = {
        0: output_color0,
        1: output_color1,
    }
    
    wall_size = 1

    is_hstack = random.Random(seed + 6).randint(0, 1) == 0
    has_wall = random.Random(seed + 7).randint(0, 1) == 0

    stack_name = 'hstack' if is_hstack else 'vstack'
    palette_name = '_samepalette' if is_same_palette_for_inputs else '_differentpalette'
    wall_name = f'_wall{wall_size}' if has_wall else ''
    task.metadata_task_id = f'{transformation_id}_{stack_name}{palette_name}{wall_name}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
            ratio_a = random.Random(iteration_seed + 5).choice(ratios)
            random_a_image = image_create_random_with_two_colors(width, height, 0, 1, ratio_a, iteration_seed + 6)
            histogram_a = Histogram.create_with_image(random_a_image)
            if histogram_a.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            ratio_b = random.Random(iteration_seed + 7).choice(ratios)
            random_b_image = image_create_random_with_two_colors(width, height, 0, 1, ratio_b, iteration_seed + 8)
            histogram_b = Histogram.create_with_image(random_b_image)
            if histogram_b.number_of_unique_colors() < 2:
                # We are not interested in empty images
                continue

            # Apply transformation
            mask_and = image_mask_and(random_a_image, random_b_image)
            mask_or = image_mask_or(random_a_image, random_b_image)
            mask_xor = image_mask_xor(random_a_image, random_b_image)

            # IDEA: allow for some ambiguity, but not too much.
            # Ensures there are no ambiguous images where the other transformations yield the same result
            ambiguous_and_or = np.array_equal(mask_and, mask_or)
            ambiguous_and_xor = np.array_equal(mask_and, mask_xor)
            is_ambiguous = ambiguous_and_or or ambiguous_and_xor
            if is_ambiguous:
                # print(f"Skipping ambiguous.")
                continue

            mask = None
            if transformation_id == 'and':
                mask = mask_and
            elif transformation_id == 'or':
                mask = mask_or
            elif transformation_id == 'xor':
                mask = mask_xor
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")
            
            # We are not interested in empty images
            histogram_mask = Histogram.create_with_image(mask)
            if histogram_mask.number_of_unique_colors() < 2:
                continue

            hstack_wall_image = image_create(wall_size, height, wall_color)
            vstack_wall_image = image_create(width, wall_size, wall_color)

            input_a_image = image_replace_colors(random_a_image, color_map_input_a)
            input_b_image = image_replace_colors(random_b_image, color_map_input_b)

            if is_hstack:
                if has_wall:
                    stacked_image = np.hstack([input_a_image, hstack_wall_image, input_b_image])
                else:
                    stacked_image = np.hstack([input_a_image, input_b_image])
            else:
                if has_wall:
                    stacked_image = np.vstack([input_a_image, vstack_wall_image, input_b_image])
                else:
                    stacked_image = np.vstack([input_a_image, input_b_image])

            input_image = stacked_image
            output_image = image_replace_colors(mask, color_map_output)
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
    transformation_ids = [
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
    seed=220000771,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
