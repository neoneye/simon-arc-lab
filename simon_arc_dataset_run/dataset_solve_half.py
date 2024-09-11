# Where is a pixel located inside an object?
# - Is it the top/bottom/left/right - half of the object, then set the mask to 1, otherwise 0.
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
from simon_arc_lab.image_trim import outer_bounding_box_after_trim_with_color
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.rectangle import Rectangle
from simon_arc_lab.image_shape3x3_center import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.rectangle_getarea import rectangle_getarea
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_half'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_half.jsonl')

def generate_task_half(seed: int, edge_name: str, connectivity: PixelConnectivity) -> Task:
    """
    Identify the mask of the pixels that are inside an object.

    Example of tasks:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=29c11459
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=639f5a19
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e7dd8335
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9ac8c9e
    """

    connectivity_name = connectivity.name.lower()

    count_example = random.Random(seed + 9).randint(4, 5)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'half_{edge_name}_{connectivity_name}'
    min_image_size = 4
    max_image_size = 6

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

            component_list = ConnectedComponent.find_objects(connectivity, random_image)
            # print(f"component_list: {component_list}")
            if len(component_list) == 0:
                continue
            accumulated_mask = np.zeros_like(random_image)
            for component in component_list:
                rect = outer_bounding_box_after_trim_with_color(component, 0)
                rect2 = rectangle_getarea(rect, edge_name)
                if rect2.is_empty():
                    continue

                # Create a mask for the rectangle.
                mask2 = np.zeros_like(component)
                x0 = rect2.x
                y0 = rect2.y
                x1 = rect2.x + rect2.width - 1
                y1 = rect2.y + rect2.height - 1
                mask2[y0:y1+1, x0:x1+1] = 1

                # Clear the area outside the rectangle.
                mask3 = np.minimum(mask2, component)

                accumulated_mask = np.maximum(accumulated_mask, mask3)

            # We are not interested in images with nothing going on.
            histogram = Histogram.create_with_image(accumulated_mask)
            if histogram.number_of_unique_colors() < 2:
                continue

            output_image = image_replace_colors(accumulated_mask, color_map_output)
            input_image = random_image
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    name_list = [
        'top',
        'bottom', 
        'left',
        'right',
    ]
    connectivity_list = [
        PixelConnectivity.ALL8,
        # PixelConnectivity.NEAREST4,
        # PixelConnectivity.CORNER4,
    ]
    accumulated_dataset_items = []
    for index_connectivity, connectivity in enumerate(connectivity_list):
        connectivity_name = connectivity.name.lower()
        for index_name, name in enumerate(name_list):
            iteration_seed = seed + 1000000 * index_name + 10000 * index_connectivity
            task = generate_task_half(iteration_seed + 1, name, connectivity)
            # task.show()
            transformation_id = f'half_{name}_{connectivity_name}'
            dataset_items = generate_dataset_item_list_inner(iteration_seed + 2, task, transformation_id)
            accumulated_dataset_items.extend(dataset_items)

    return accumulated_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1319377377,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
