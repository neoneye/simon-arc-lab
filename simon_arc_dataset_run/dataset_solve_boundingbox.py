# Identify the bounding boxes of objects in the image.
#
# IDEA: show one object
# IDEA: show multiple objects
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
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.image_rect import image_rect
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_trim import outer_bounding_box_after_trim_with_color
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_boundingbox'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_boundingbox.jsonl')

def generate_task_boundingbox(seed: int, transformation_id: str) -> Task:
    """
    Show one object in the image, and identify the bounding box of the object.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = transformation_id
    min_image_size = 3
    max_image_size = 5

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)

    color_map_input_a = {
        0: input_colors[0],
        1: input_colors[1],
    }
    color_map_input_b = {
        0: input_colors[2],
        1: input_colors[3],
    }

    connectivity = PixelConnectivity.ALL8

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(output_colors)
    output_color0 = output_colors[0]
    output_color1 = output_colors[1]
    color_map_output = {
        0: output_color0,
        1: output_color1,
    }

    task.metadata_task_id = f'{transformation_id}'
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            parent_rect = Rectangle(0, 0, width, height)
            object_a_rect = parent_rect.random_child_rectangle(iteration_seed + 3)
            if object_a_rect.mass() < 2:
                # We are not interested in empty or 1px images
                continue

            # background image with two colors
            # ratio_background = random.Random(iteration_seed + 5).choice(ratios)
            # background_image = image_create_random_with_two_colors(width, height, 0, 1, ratio_background, iteration_seed + 6)
            background_image = image_create(width, height, 0)
            # histogram_background = Histogram.create_with_image(background_image)
            # if histogram_background.number_of_unique_colors() < 2:
            #     # We are not interested in empty images
            #     continue

            # object A, with two colors
            ratio_b = random.Random(iteration_seed + 7).choice(ratios)
            random_b_image = image_create_random_with_two_colors(width, height, 0, 1, ratio_b, iteration_seed + 8)

            mask_a = image_create(width, height, 0)
            mask_a = image_rect(mask_a, object_a_rect, 1)

            # multiply the mask with the object image
            mixed_image = image_mix(mask_a, background_image, random_b_image)

            component_list = ConnectedComponent.find_objects_with_ignore_mask_inner(connectivity, mixed_image, background_image)
            # print(f"component_list: {component_list}")
            if len(component_list) == 0:
                continue

            found_component = None
            found_mass = 0
            for component in component_list:
                if component.mass > found_mass:
                    found_mass = component.mass
                    found_component = component
                    break

            if found_component is None:
                continue

            if found_mass < 2:
                continue

            bounding_box = outer_bounding_box_after_trim_with_color(found_component.mask, 1)
            mask_b = image_create(width, height, 0)
            mask_b = image_rect(mask_b, bounding_box, 1)

            # input_image = mixed_image
            # input_image = image_replace_colors(mixed_image, color_map_output)
            input_image = found_component.mask
            # output_image = image_replace_colors(mask_a, color_map_output)
            output_image = mask_b
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(5):
        task = generate_task_boundingbox(i, 'one_object')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_ids = [
        'one_object',
    ]
    accumulated_dataset_items = []
    for index, transformation_id in enumerate(transformation_ids):
        task = generate_task_boundingbox(seed + index * 100338383, transformation_id)
        task.show()
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
