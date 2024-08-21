# Grid transformations.
# - Extract content from an grid with irregular sized cells.
# - Mutate the content inside the grid: flipx, flipy, rotate 180 degrees.
# 
# IDEA: Add noise to the input grid cells. Either the most/least popular color of the cell, becomes the output color.
# IDEA: Convert the input to a grid.
# IDEA: different colors for grid horizontal/vertical lines, and grid line intersections.
# IDEA: varying grid line sizes.
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
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.task import *
from simon_arc_lab.image_rect import *
from simon_arc_lab.image_grid import ImageGridBuilder
from simon_arc_lab.image_util import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_grid'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_grid.jsonl')

def generate_task_extract_content_from_grid(seed: int) -> Task:
    """
    Original image is random noise.
    Wrap the original image in a grid in different ways.
    The job is to extract the original image from the grid structure.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 5

    min_cell_size = 1
    max_cell_size = 6

    # grid colors
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_mapping = {}
    for i, color in enumerate(colors):
        color_mapping[i] = color

    color_grid = 0
    color_deceiving_pixel = color_grid

    color_mapping_eliminate_grid_color = {
        0: 1,
    }

    has_no_top_line = random.Random(seed + 4).randint(0, 99) > 50
    has_no_bottom_line = random.Random(seed + 5).randint(0, 99) > 50
    has_no_left_line = random.Random(seed + 6).randint(0, 99) > 50
    has_no_right_line = random.Random(seed + 7).randint(0, 99) > 50

    grid_size = random.Random(seed + 8).randint(1, 2)

    task.metadata_task_id = 'extract_content_from_grid'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            seed_for_input_image_data = (retry_index * 10000) + ((seed + 17) * 37) + 101 + i * 1991

            candidate_image = image_create_random_advanced(seed_for_input_image_data + 1, min_image_size, max_image_size, min_image_size, max_image_size)
            height, width = candidate_image.shape
            candidate_image = image_replace_colors(candidate_image, color_mapping_eliminate_grid_color)

            # The image has one pixel with the same color as the grid color.
            has_deceiving_pixel = random.Random(seed_for_input_image_data + 2).randint(0, 99) > 30
            if has_deceiving_pixel:
                rx = random.Random(seed_for_input_image_data + 1).randint(0, width - 1)
                ry = random.Random(seed_for_input_image_data + 2).randint(0, height - 1)
                candidate_image[ry, rx] = color_deceiving_pixel

            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue

            builder = ImageGridBuilder(width, height)
            builder.set_cell_size_random(seed_for_input_image_data * 3779 + 188282821, min_cell_size, max_cell_size)
            builder.set_separator_size(grid_size)

            if has_no_top_line:
                builder.set_top_separator_size(0)
            
            if has_no_bottom_line:
                builder.set_bottom_separator_size(0)

            if has_no_left_line:
                builder.set_left_separator_size(0)

            if has_no_right_line:
                builder.set_right_separator_size(0)

            grid_image = builder.draw(candidate_image, color_grid)

            input_image = image_replace_colors(grid_image, color_mapping)
            output_image = image_replace_colors(candidate_image, color_mapping)
            break
        if input_image is None or output_image is None:
            raise Exception("Failed to create a pair.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_mutate_content_inside_grid(seed: int, transformation_id: str) -> Task:
    """
    Original image is random noise.
    Wrap the original image in a grid in different ways.
    The job is to manipulate the image, while preserving the grid structure.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    min_cell_size = 1
    max_cell_size = 3

    # grid colors
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_mapping = {}
    for i, color in enumerate(colors):
        color_mapping[i] = color

    color_grid = 0
    color_deceiving_pixel = color_grid

    color_mapping_eliminate_grid_color = {
        0: 1,
    }

    has_no_top_line = random.Random(seed + 4).randint(0, 99) > 50
    has_no_bottom_line = random.Random(seed + 5).randint(0, 99) > 50
    has_no_left_line = random.Random(seed + 6).randint(0, 99) > 50
    has_no_right_line = random.Random(seed + 7).randint(0, 99) > 50

    grid_size = random.Random(seed + 8).randint(1, 2)

    task.metadata_task_id = f'mutate_content_inside_grid_{transformation_id}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            seed_for_input_image_data = (retry_index * 10000) + ((seed + 17) * 37) + 101 + i * 1991

            candidate_image = image_create_random_advanced(seed_for_input_image_data + 1, min_image_size, max_image_size, min_image_size, max_image_size)
            height, width = candidate_image.shape
            candidate_image = image_replace_colors(candidate_image, color_mapping_eliminate_grid_color)

            # The image has one pixel with the same color as the grid color.
            has_deceiving_pixel = random.Random(seed_for_input_image_data + 2).randint(0, 99) > 30
            if has_deceiving_pixel:
                rx = random.Random(seed_for_input_image_data + 1).randint(0, width - 1)
                ry = random.Random(seed_for_input_image_data + 2).randint(0, height - 1)
                candidate_image[ry, rx] = color_deceiving_pixel

            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue

            builder = ImageGridBuilder(width, height)
            builder.set_cell_size_random(seed_for_input_image_data * 3779 + 188282821, min_cell_size, max_cell_size)
            builder.set_separator_size(grid_size)

            if has_no_top_line:
                builder.set_top_separator_size(0)
            
            if has_no_bottom_line:
                builder.set_bottom_separator_size(0)

            if has_no_left_line:
                builder.set_left_separator_size(0)

            if has_no_right_line:
                builder.set_right_separator_size(0)

            grid_input_image = builder.draw(candidate_image, color_grid)

            if transformation_id == 'flipx':
                modified_image = image_flipx(candidate_image)
            elif transformation_id == 'flipy':
                modified_image = image_flipy(candidate_image)
            elif transformation_id == '180':
                modified_image = image_rotate_180(candidate_image)
            else:
                raise ValueError(f"Unknown transformation id: {transformation_id}")

            grid_output_image = builder.draw(modified_image, color_grid)

            input_image = image_replace_colors(grid_input_image, color_mapping)
            output_image = image_replace_colors(grid_output_image, color_mapping)
            break
        if input_image is None or output_image is None:
            raise Exception("Failed to create a pair.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(100):
        j = i % 4
        if j == 0:
            task = generate_task_extract_content_from_grid(i)
        elif j == 1:
            task = generate_task_mutate_content_inside_grid(i, 'flipx')
        elif j == 2:
            task = generate_task_mutate_content_inside_grid(i, 'flipy')
        elif j == 3:
            task = generate_task_mutate_content_inside_grid(i, '180')
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
        task = generate_task_extract_content_from_grid(seed)
        transformation_id = 'extract_content_from_grid'
    elif j == 1:
        task = generate_task_mutate_content_inside_grid(seed, 'flipx')
        transformation_id = 'mutate_content_inside_grid_flipx'
    elif j == 2:
        task = generate_task_mutate_content_inside_grid(seed, 'flipy')
        transformation_id = 'mutate_content_inside_grid_flipy'
    elif j == 3:
        task = generate_task_mutate_content_inside_grid(seed, '180')
        transformation_id = 'mutate_content_inside_grid_180'

    # task.show()
    items = generate_dataset_item_list_inner((seed + 1) * 11, task, transformation_id)
    return items


generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=51300013,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
