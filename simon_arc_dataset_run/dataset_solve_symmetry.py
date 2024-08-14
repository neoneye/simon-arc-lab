# Symmetric transformations.
# - generate a palindrome image, hstack(original, flipx(original))
# - extract the tile that makes up the symmetric input image.
#
# IDEA: Use rotate cw/ccw to create a symmetric image.
# IDEA: Use flip diagonal to create a symmetric image.
# IDEA: Introduce hstack4, hstack5, vstack4, vstack5.
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
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_symmetry'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_symmetry.jsonl')

def generate_task_with_input_image_create_output_symmetry(seed: int) -> Task:
    """
    Create a symmetric image from an input image.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 2
    max_width = 4
    min_height = 2
    max_height = 4

    image_symmetry = ImageSymmetry.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)
    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = instruction_sequence

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced((seed * 31) + 1002 + i, min_width, max_width, min_height, max_height)
        output_image = image_symmetry.execute(input_image)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetric_input_image_and_extract_a_particular_tile(seed: int) -> Task:
    """
    Identify the top-left tile of a symmetric image.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 2
    max_width = 3
    min_height = 2
    max_height = 3

    image_symmetry = ImageSymmetry.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    # It's the top-left tile that is always extracted. It's the first tile.
    image_symmetry.use_original_for_index(0)

    instruction_sequence = image_symmetry.instruction_sequence()
    task.metadata_task_id = instruction_sequence

    # Rotate the image, so the tile is not always at the top-left, but also in the other corners.
    rotate90_count = random.Random(seed + 3).randint(0, 3)
    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image_raw = image_create_random_advanced((seed * 31) + 1002 + i, min_width, max_width, min_height, max_height)
        input_image_raw = image_symmetry.execute(output_image_raw)
        input_image = np.rot90(input_image_raw, rotate90_count)
        output_image = np.rot90(output_image_raw, rotate90_count)
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        if i % 2 == 0:
            task = generate_task_with_input_image_create_output_symmetry(i)
        else:
            task = generate_task_with_symmetric_input_image_and_extract_a_particular_tile(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    if seed % 2 == 0:
        task = generate_task_with_input_image_create_output_symmetry(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'create_symmetry {task_id}'"
    else:
        task = generate_task_with_symmetric_input_image_and_extract_a_particular_tile(seed)
        task_id = task.metadata_task_id
        transformation_id = f"'extract_tile {task_id}'"
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=218000410,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
