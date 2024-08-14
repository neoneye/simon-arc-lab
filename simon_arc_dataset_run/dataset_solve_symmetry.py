# Symmetric transformations.
# - generate a palindrome image, hstack(original, flipx(original))
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
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 2
    max_width = 3
    min_height = 2
    max_height = 3

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

def demo_generate_task():
    for i in range(5):
        task = generate_task_with_input_image_create_output_symmetry(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    all_dataset_items = []
    task = generate_task_with_input_image_create_output_symmetry(seed)
    task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, 'create_symmetry')
    all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=118000410,
    max_num_samples=10,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
