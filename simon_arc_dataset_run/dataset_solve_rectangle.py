# Do things with rectangles.
#
# IDEA: invert hollow/filled rectangles.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4347f46a
#
# IDEA: fill the interior with another color.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=50cb2852
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d5d6de2d
#
# IDEA: fill the interior with colors depending on the mass.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=694f12f3
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
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_rect import image_rect, image_rect_hollow
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_paste import *
from simon_arc_lab.rectangle import Rectangle
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_rectangle'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_rectangle.jsonl')

def generate_task_with_rectangles(seed: int, transformation_id: str) -> Task:
    """
    Transformations with a few non-overlapping rectangles.
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    task = Task()
    min_rect_size = 3
    max_rect_size = 5
    min_image_size = 8
    max_image_size = 14
    min_rect_count = 2
    max_rect_count = 3

    color_background = 0
    color_rect = 1

    available_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(available_colors)
    color_map = {}
    for i in range(10):
        color_map[i] = available_colors[i]

    task.metadata_task_id = f"rectangle {transformation_id}"

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_data = seed * 6 + retry_index * 137 + i

            width = random.Random(seed_for_input_image_data + 100).randint(min_image_size, max_image_size)
            height = random.Random(seed_for_input_image_data + 101).randint(min_image_size, max_image_size)

            rect_count = random.Random(seed_for_input_image_data + 102).randint(min_rect_count, max_rect_count)
            rects = []
            for j in range(rect_count):
                for retry_index2 in range(100):
                    r_width = random.Random(seed_for_input_image_data + retry_index2 + j + 100).randint(min_rect_size, max_rect_size)
                    r_height = random.Random(seed_for_input_image_data + retry_index2 + j + 101).randint(min_rect_size, max_rect_size)
                    r_x = random.Random(seed_for_input_image_data + retry_index2 + j + 102).randint(0, width - r_width - 1)
                    r_y = random.Random(seed_for_input_image_data + retry_index2 + j + 103).randint(0, height - r_height - 1)
                    rect = Rectangle(r_x, r_y, r_width, r_height)

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

            result_input_image = image_create(width, height, color_background)
            result_output_image = image_create(width, height, color_background)
            if transformation_id == "fill_the_hollow_rectangles":
                for rect_index, rect in enumerate(rects):
                    result_input_image = image_rect_hollow(result_input_image, rect, color_rect, 1)
                    result_output_image = image_rect(result_output_image, rect, color_rect)
            elif transformation_id == "hollow_out_the_filled_rectangles":
                for rect_index, rect in enumerate(rects):
                    result_input_image = image_rect(result_input_image, rect, color_rect)
                    result_output_image = image_rect_hollow(result_output_image, rect, color_rect, 1)
            elif transformation_id == "mask_of_the_hollow_areas":
                for rect_index, rect in enumerate(rects):
                    result_input_image = image_rect_hollow(result_input_image, rect, color_rect, 1)
                    rect2 = Rectangle(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)
                    result_output_image = image_rect(result_output_image, rect2, color_rect)
            elif transformation_id == "mask_outside_the_hollow_rectangles":
                for rect_index, rect in enumerate(rects):
                    result_input_image = image_rect_hollow(result_input_image, rect, color_rect, 1)
                    rect2 = Rectangle(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
                    result_output_image = image_rect_hollow(result_output_image, rect2, color_rect, 1)
            elif transformation_id == "mask_outside_the_filled_rectangles":
                for rect_index, rect in enumerate(rects):
                    result_input_image = image_rect(result_input_image, rect, color_rect)
                    rect2 = Rectangle(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
                    result_output_image = image_rect_hollow(result_output_image, rect2, color_rect, 1)
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
    j = seed % 5
    # j = 4
    if j == 0:
        task = generate_task_with_rectangles(seed, "fill_the_hollow_rectangles")
    elif j == 1:
        task = generate_task_with_rectangles(seed, "hollow_out_the_filled_rectangles")
    elif j == 2:
        task = generate_task_with_rectangles(seed, "mask_of_the_hollow_areas")
    elif j == 3:
        task = generate_task_with_rectangles(seed, "mask_outside_the_hollow_rectangles")
    elif j == 4:
        task = generate_task_with_rectangles(seed, "mask_outside_the_filled_rectangles")
    # task.show()
    transformation_id = task.metadata_task_id
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=7382000031,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
