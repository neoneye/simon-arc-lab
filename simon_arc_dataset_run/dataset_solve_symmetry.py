# Symmetric transformations.
# - generate a palindrome image, hstack(original, flipx(original))
# - extract the tile that makes up the symmetric input image.
# - Use rotate cw/ccw to create a symmetric image.
# - Use flip diagonal to create a symmetric image.
#
# IDEA: rotational symmetry, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2697da3f
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=46442a0e
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7fe24cdd
#
# IDEA: mirror the image, where the mirrored version has a different color scheme.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6f473927
#
# IDEA: Extract tile that is repeated in the symmetric image, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2013d3e2
#
# IDEA: repair rotational symmetry, with red color, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1b60fb0c
#
# IDEA: currently extract a tile from a corner. Also extract the tile from the center, or another coordinate.
# IDEA: Introduce grid2x3, grid3x3, grid3x3.
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
from simon_arc_lab.image_symmetry import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_symmetry'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_symmetry.jsonl')

def generate_task_with_input_image_create_output_symmetry_rect(seed: int) -> Task:
    """
    Create a symmetric image from a rectangular input image.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    image_symmetry = ImageSymmetryRect.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)
    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = "rect " + instruction_sequence

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced((seed * 31) + 1002 + i, min_image_size, max_image_size, min_image_size, max_image_size)
        output_image = image_symmetry.execute(input_image)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_input_image_create_output_symmetry_square(seed: int) -> Task:
    """
    Create a symmetric image from a square input image.
    """
    count_example = random.Random(seed + 9).randint(3, 5)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    image_symmetry = ImageSymmetrySquare.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)
    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = "square " + instruction_sequence

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_size = seed * 5 + retry_index * 133 + i
            seed_for_input_image_data = seed * 6 + retry_index * 137 + i
            image_size = random.Random(seed_for_input_image_size).randint(min_image_size, max_image_size)
            candidate_image = image_create_random_advanced(seed_for_input_image_data, image_size, image_size, image_size, image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() > 1:
                input_image = candidate_image
                break
        if input_image is None:
            raise Exception("Failed to create a square image with more than one color")
        output_image = image_symmetry.execute(input_image)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetry_rect_input_image_and_extract_a_particular_tile(seed: int) -> Task:
    """
    Identify the top-left rectangular tile of a symmetric image.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    image_symmetry = ImageSymmetryRect.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    # It's the top-left tile that is always extracted. It's the first tile.
    image_symmetry.use_mutation_for_index(0, ImageSymmetryMutationId.ORIGINAL)

    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = "rect " + instruction_sequence

    # Rotate the image, so the tile is not always at the top-left, but also in the other corners.
    rotate90_count = random.Random(seed + 3).randint(0, 3)
    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image_raw = image_create_random_advanced((seed * 31) + 1002 + i, min_image_size, max_image_size, min_image_size, max_image_size)
        input_image_raw = image_symmetry.execute(output_image_raw)
        input_image = np.rot90(input_image_raw, rotate90_count)
        output_image = np.rot90(output_image_raw, rotate90_count)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(seed: int) -> Task:
    """
    Identify the top-left square tile of a symmetric image.
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    image_symmetry = ImageSymmetryRect.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    # It's the top-left tile that is always extracted. It's the first tile.
    image_symmetry.use_mutation_for_index(0, ImageSymmetryMutationId.ORIGINAL)

    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = "square " + instruction_sequence

    # Rotate the image, so the tile is not always at the top-left, but also in the other corners.
    rotate90_count = random.Random(seed + 3).randint(0, 3)
    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image_raw = None
        for retry_index in range(100):
            seed_for_input_image_size = seed * 5 + retry_index * 133 + i
            seed_for_input_image_data = seed * 7 + retry_index * 133 + i
            image_size = random.Random(seed_for_input_image_size).randint(min_image_size, max_image_size)
            candidate_image = image_create_random_advanced(seed_for_input_image_data, image_size, image_size, image_size, image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() > 1:
                output_image_raw = candidate_image
                break
        if output_image_raw is None:
            raise Exception("Failed to create a square image with more than one color")
        input_image_raw = image_symmetry.execute(output_image_raw)
        input_image = np.rot90(input_image_raw, rotate90_count)
        output_image = np.rot90(output_image_raw, rotate90_count)
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        j = i % 4
        task = generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(i)
        if j == 0:
            task = generate_task_with_input_image_create_output_symmetry_rect(i)
        elif j == 1:
            task = generate_task_with_symmetry_rect_input_image_and_extract_a_particular_tile(i)
        elif j == 2:
            task = generate_task_with_input_image_create_output_symmetry_square(i)
        elif j == 3:
            task = generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 4
    if j == 0:
        task = generate_task_with_input_image_create_output_symmetry_rect(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'create_rect_symmetry {task_id}'"
    elif j == 1:
        task = generate_task_with_symmetry_rect_input_image_and_extract_a_particular_tile(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'extract_rect_tile {task_id}'"
    elif j == 2:
        task = generate_task_with_input_image_create_output_symmetry_square(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'create_square_symmetry {task_id}'"
    elif j == 3:
        task = generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'extract_square_tile {task_id}'"
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=718000410,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
