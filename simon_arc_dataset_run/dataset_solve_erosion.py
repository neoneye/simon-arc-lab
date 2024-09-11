# Erosion transformations.
# - remove the outer most pixel from objects, and return a mask with what remains.
#
# IDEA: Extrude object in different directions up/down/left/right with a background color indicating the mask of available space.
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
from simon_arc_lab.image_erosion_multicolor import *
from simon_arc_lab.image_util import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_erosion'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_erosion.jsonl')

def generate_task_erosion(seed: int, connectivity: PixelConnectivity) -> Task:
    """
    Create an eroded image from an input image, by removing the outermost pixels.
    """
    connectivity_name_lower = connectivity.name.lower()

    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'erosion {connectivity_name_lower}'
    min_width = 3
    max_width = 8
    min_height = 3
    max_height = 8

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 11).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color_mapping = {
        0: color0,
        1: color1,
    }

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            input_image = image_create_random_advanced((retry_index * 10000) + (seed * 37) + 101 + i, min_width, max_width, min_height, max_height)
            image_mask = image_erosion_multicolor(input_image, connectivity)

            # most of the eroded images, have the same color for all pixels, except the border. Ignore those.
            # I'm only interested in non-trivial images where there is stuff remaining after the erosion has taken place.
            # remove the 1px border around the image
            image_mask_without_border = image_mask[1:-1, 1:-1]
            count_ones = np.count_nonzero(image_mask_without_border == 1)
            count_zeros = np.count_nonzero(image_mask_without_border == 0)
            if count_ones == 0 or count_zeros == 0:
                # there is nothing remaining after the erosion, skip this image.
                continue

            # loop through all the connectivity enums, and check if the image is the same as the input image.
            # if it is, then skip this image, since I don't want a puzzle that is ambiguous.
            for pc in PixelConnectivity:
                if pc == connectivity:
                    continue
                image_mask_other = image_erosion_multicolor(input_image, pc)
                if np.array_equal(image_mask, image_mask_other):
                    continue

            output_image = image_replace_colors(image_mask, color_mapping)
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    connectivity_list = [
        PixelConnectivity.NEAREST4,
        PixelConnectivity.ALL8,
        PixelConnectivity.CORNER4,
        PixelConnectivity.LR2,
        PixelConnectivity.TB2,
        PixelConnectivity.TLBR2,
        PixelConnectivity.TRBL2,
    ]
    accumulated_items = []
    for index, connectivity in enumerate(connectivity_list):
        connectivity_name_lower = connectivity.name.lower()
        transformation_id = f'apply_erosion_{connectivity_name_lower}'

        for retry_index in range(20):
            current_seed = seed + index * 10000 + 1333 * retry_index
            try:
                task = generate_task_erosion(current_seed, connectivity)
                # task.show()
                items = generate_dataset_item_list_inner(seed, task, transformation_id)
                accumulated_items.extend(items)
                break
            except Exception as e:
                print(f"trying again {retry_index} with connectivity {connectivity}. error: {e}")
    if len(accumulated_items) == 0:
        print(f"Failed to generate any dataset items")
    return accumulated_items


generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=613600313,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
