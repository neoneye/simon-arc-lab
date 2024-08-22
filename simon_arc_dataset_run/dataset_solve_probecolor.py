# probe color directional transformation:
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
from simon_arc_lab.image_raytrace_probecolor import ImageRaytraceProbeColorDirection, image_raytrace_probecolor_direction
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_probecolor'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_probecolor.jsonl')

def generate_task(seed: int) -> Task:
    random.seed(seed)

    count_example = random.randint(3, 4)
    count_test = random.randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 9

    probecolor_direction_list = [
        ImageRaytraceProbeColorDirection.TOP,
        ImageRaytraceProbeColorDirection.BOTTOM,
        ImageRaytraceProbeColorDirection.LEFT,
        ImageRaytraceProbeColorDirection.RIGHT,
        ImageRaytraceProbeColorDirection.TOPLEFT,
        ImageRaytraceProbeColorDirection.TOPRIGHT,
        ImageRaytraceProbeColorDirection.BOTTOMLEFT,
        ImageRaytraceProbeColorDirection.BOTTOMRIGHT,
    ]
    probe_color_direction = random.choice(probecolor_direction_list)
    direction_name = probe_color_direction.name.lower()

    task.metadata_task_id = f'probecolor_{direction_name}'

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)
    color_edge = colors[0]
    color_other = colors[1]
    color_map_eliminate_edge_color = {
        color_edge: color_other,
    }
    for i in range(count_example+count_test):
        is_example = i < count_example
        if is_example == False:
            min_image_size = 1
            max_image_size = 14

        random_image = None
        for retry_index in range(100):
            image_seed = seed * 1003 + i * 112 + 22828 + retry_index * 113131
            random_image = image_create_random_advanced(image_seed, min_image_size, max_image_size, min_image_size, max_image_size)
            histogram = Histogram.create_with_image(random_image)
            if histogram.number_of_unique_colors() < 2:
                # 2 or more colors must be present in the image
                continue
            # there are 2 or more colors, so stop the loop
            break
        if random_image is None:
            raise ValueError("Failed to create a random image")
        
        random_image_without_edgecolor = image_replace_colors(random_image, color_map_eliminate_edge_color)
        output_image = image_raytrace_probecolor_direction(random_image_without_edgecolor, color_edge, probe_color_direction)

        input_image = random_image_without_edgecolor
        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(10):
        task = generate_task(i)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    task = generate_task(seed)
    transformation_id = task.metadata_task_id
    # task.show()
    items = generate_dataset_item_list_inner(seed * 101 + 9393, task, transformation_id)
    return items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=354300232,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
