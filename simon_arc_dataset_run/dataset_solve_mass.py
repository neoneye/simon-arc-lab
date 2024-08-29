# Identify areas with a particular mass.
#
# IDEA: train with other connectitivity types, than ALL8, so that a task like this can be solved.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=aedd82e4
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
from simon_arc_lab.image_create_random_simple import image_create_random_with_two_colors
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_object_mass import *
from simon_arc_lab.image_mass_compare import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_mass'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_mass.jsonl')

def generate_task_specific_mass(seed: int, find_mass_size: int, connectivity: PixelConnectivity) -> Task:
    """
    Identify the areas with a particular mass.

    Example of tasks with mass=1:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=178fcbfb
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=23581191
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2dc579da
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3f23242b
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4258a5f9
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=444801d8
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=56ff96f3
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8d510a79
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8403a5d5
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=99fa7670
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=dc1df850
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ddf7fa4f
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ded97339
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e179c5f4
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9614598
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ea786f4a
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ecdecbb3
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f15e1fac

    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'mass{find_mass_size}_{connectivity.name.lower()}'
    min_image_size = 1
    max_image_size = 8

    connectivity = PixelConnectivity.ALL8

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_map_output = {
        0: color0,
        1: color1,
    }
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101
            random_image = image_create_random_advanced(iteration_seed, min_image_size, max_image_size, min_image_size, max_image_size)
            height, width = random_image.shape

            component_list = ConnectedComponent.find_objects(connectivity, random_image)
            # print(f"component_list: {component_list}")
            if len(component_list) == 0:
                continue

            # Identify the lonely pixels
            mass_image = object_mass(component_list)
            mask = np.zeros_like(mass_image)
            for y in range(height):
                for x in range(width):
                    mass = mass_image[y, x]
                    if mass != find_mass_size:
                        continue
                    mask[y, x] = 1

            # We are not interested in images with zero lonely pixels
            histogram = Histogram.create_with_image(mask)
            if histogram.number_of_unique_colors() < 2:
                continue

            output_image = image_replace_colors(mask, color_map_output)
            input_image = random_image
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_comparing_adjacent_rowcolumn(seed: int, transformation_id: str) -> Task:
    """
    Identify the areas with a particular mass.
    """
    count_example = random.Random(seed + 9).randint(3, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'mass_compare_{transformation_id}'
    min_image_size = 4
    max_image_size = 7

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101
            random_image = image_create_random_advanced(iteration_seed, min_image_size, max_image_size, min_image_size, max_image_size)

            if transformation_id == 'adjacent_rows':
                output_image = image_mass_compare_adjacent_rows(random_image, color0, color1, color2)
            elif transformation_id == 'adjacent_columns':
                output_image = image_mass_compare_adjacent_columns(random_image, color0, color1, color2)
            else:
                raise ValueError(f'Unknown transformation_id: {transformation_id}')

            # We are not interested in images with zero lonely pixels
            histogram = Histogram.create_with_image(output_image)
            if histogram.number_of_unique_colors() < 3:
                continue

            input_image = random_image
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_specific_mass(i, 1, PixelConnectivity.ALL8)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 6
    if j == 0:
        transformation_id = 'mass1_all8'
        task = generate_task_specific_mass(seed, 1, PixelConnectivity.ALL8)
    elif j == 1:
        transformation_id = 'mass2_all8'
        task = generate_task_specific_mass(seed, 2, PixelConnectivity.ALL8)
    elif j == 2:
        transformation_id = 'mass3_all8'
        task = generate_task_specific_mass(seed, 3, PixelConnectivity.ALL8)
    elif j == 3:
        transformation_id = 'mass4_all8'
        task = generate_task_specific_mass(seed, 4, PixelConnectivity.ALL8)
    elif j == 4:
        transformation_id = 'mass_compare_adjacent_rows'
        task = generate_task_comparing_adjacent_rowcolumn(seed, 'adjacent_rows')
    elif j == 5:
        transformation_id = 'mass_compare_adjacent_columns'
        task = generate_task_comparing_adjacent_rowcolumn(seed, 'adjacent_columns')
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=170000777,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
