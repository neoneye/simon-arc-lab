# Insert objects into templates
#
# IDEA: Template with flipx/flipy.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9f27f097
#
# IDEA: Templates with non-rectangular shapes.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=321b1fc6
#
# IDEA: Templates with a marker.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=88a10436
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
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_paste import *
from simon_arc_lab.find_bounding_box import find_bounding_box_ignoring_color
from simon_arc_lab.rectangle import Rectangle
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_template'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_template.jsonl')

def generate_task_with_template_areas(seed: int, transformation_id: str) -> Task:
    """
    Create some template areas, and insert objects into them.

    Example:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e76a88a6
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    task = Task()
    min_template_size = 2
    max_template_size = 3
    min_image_size = 6
    max_image_size = 10
    min_rect_count = 2
    max_rect_count = 3

    color_background = 0
    color_template = 1

    available_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(available_colors)
    color_map = {}
    for i in range(10):
        color_map[i] = available_colors[i]

    task.metadata_task_id = f"template {transformation_id}"

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_data = seed * 6 + retry_index * 137 + i
            insertion_image = image_create_random_advanced(seed_for_input_image_data, min_template_size, max_template_size, min_template_size, max_template_size)
            histogram = Histogram.create_with_image(insertion_image)
            if histogram.number_of_unique_colors() < 2:
                continue
            insertion_image_height, insertion_image_width = insertion_image.shape

            bounding_box_background = find_bounding_box_ignoring_color(insertion_image, color_background)
            if bounding_box_background.width != insertion_image_width or bounding_box_background.height != insertion_image_height:
                # print("skip image with bounding box not matching the image size, background color")
                continue
            bounding_box_template = find_bounding_box_ignoring_color(insertion_image, color_template)
            if bounding_box_template.width != insertion_image_width or bounding_box_template.height != insertion_image_height:
                # print("skip image with bounding box not matching the image size, template color")
                continue

            width = random.Random(seed_for_input_image_data + 100).randint(min_image_size, max_image_size)
            height = random.Random(seed_for_input_image_data + 101).randint(min_image_size, max_image_size)
            result_input_image = image_create(width, height, color_background)
            result_output_image = image_create(width, height, color_background)

            rect_count = random.Random(seed_for_input_image_data + 102).randint(min_rect_count, max_rect_count)
            rects = []
            for j in range(rect_count):
                for retry_index2 in range(100):
                    rect = rectangle_for_random_paste(insertion_image, result_output_image, seed_for_input_image_data + 3 + j * 130 + retry_index2 * 1000)
                    rect_padded = Rectangle(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
                    good_rect = True
                    for r in rects:
                        if r.has_overlap(rect_padded):
                            good_rect = False
                            break
                    if good_rect:
                        rects.append(rect)
                        break

            if len(rects) != rect_count:
                # print("unable to find non-overlapping rectangles")
                continue

            template_image = image_create(insertion_image_width, insertion_image_height, color_template)

            if transformation_id == "with_insertion_image":
                for rect_index, rect in enumerate(rects):
                    result_output_image = image_paste_at(insertion_image, result_output_image, rect.x, rect.y)

                    if rect_index > 0:
                        result_input_image = image_paste_at(template_image, result_input_image, rect.x, rect.y)
                    else:
                        result_input_image = image_paste_at(insertion_image, result_input_image, rect.x, rect.y)
            elif transformation_id == "without_insertion_image":
                for rect_index, rect in enumerate(rects):
                    if rect_index > 0:
                        result_output_image = image_paste_at(insertion_image, result_output_image, rect.x, rect.y)

                    if rect_index > 0:
                        result_input_image = image_paste_at(template_image, result_input_image, rect.x, rect.y)
                    else:
                        result_input_image = image_paste_at(insertion_image, result_input_image, rect.x, rect.y)
            elif transformation_id == "swap_one_to_many":
                for rect_index, rect in enumerate(rects):
                    if rect_index > 0:
                        result_output_image = image_paste_at(insertion_image, result_output_image, rect.x, rect.y)
                    else:
                        result_output_image = image_paste_at(template_image, result_output_image, rect.x, rect.y)

                    if rect_index > 0:
                        result_input_image = image_paste_at(template_image, result_input_image, rect.x, rect.y)
                    else:
                        result_input_image = image_paste_at(insertion_image, result_input_image, rect.x, rect.y)
            elif transformation_id == "swap_many_to_one":
                for rect_index, rect in enumerate(rects):
                    if rect_index > 0:
                        result_output_image = image_paste_at(template_image, result_output_image, rect.x, rect.y)
                    else:
                        result_output_image = image_paste_at(insertion_image, result_output_image, rect.x, rect.y)

                    if rect_index > 0:
                        result_input_image = image_paste_at(insertion_image, result_input_image, rect.x, rect.y)
                    else:
                        result_input_image = image_paste_at(template_image, result_input_image, rect.x, rect.y)
            else:
                raise ValueError(f"Unknown transformation_id: {transformation_id}")


            # Palette
            input_image = image_replace_colors(result_input_image, color_map)
            output_image = image_replace_colors(result_output_image, color_map)
            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 4
    j = (seed % 2) + 2
    j = 2
    if j == 0:
        task = generate_task_with_template_areas(seed, "with_insertion_image")
    elif j == 1:
        task = generate_task_with_template_areas(seed, "without_insertion_image")
    elif j == 2:
        task = generate_task_with_template_areas(seed, "swap_one_to_many")
    else:
        task = generate_task_with_template_areas(seed, "swap_many_to_one")
    # task.show()
    transformation_id = task.metadata_task_id
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=7385600041,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
