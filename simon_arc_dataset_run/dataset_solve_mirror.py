# mirror transformations:
# - Move objects to the other side of the image.
#
# IDEA: Swap objects with their mirror image.
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
from simon_arc_lab.image_pad import *
from simon_arc_lab.image_paste import *
from simon_arc_lab.find_bounding_box import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.cellular_automaton import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_mirror'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_mirror.jsonl')

MAX_IMAGE_SIZE = 12

def random_asymmetric_object_maze(seed: int, min_size: int, max_size: int, color0: int, color1: int) -> np.array:
    if min_size > max_size:
        raise Exception(f"min_size={min_size} > max_size={max_size}")
    
    for retry_index in range(100):
        iteration_seed = seed * 37 + retry_index * 100
        width = random.Random(iteration_seed + 1).randint(min_size, max_size)
        height = random.Random(iteration_seed + 2).randint(min_size, max_size)

        ratios = [0.4, 0.5, 0.6]
        ratio = random.Random(iteration_seed + 3).choice(ratios)
        random_image_raw = image_create_random_with_two_colors(width, height, 0, 1, ratio, iteration_seed + 4)
        random_image01 = CARuleMaze().apply_wrap(random_image_raw, wrapx=False, wrapy=False, outside_value=0, step_count=3)

        histogram = Histogram.create_with_image(random_image01)
        if histogram.number_of_unique_colors() != 2:
            # Avoid having an empty image
            continue

        count0 = histogram.get_count_for_color(0)
        count1 = histogram.get_count_for_color(1)
        total = count0 + count1
        ratio = count0 / total
        if ratio < 0.2 or ratio > 0.8:
            # print(f"Avoid having a single color dominating the image. count0={count0} count1={count1} total={total} ratio={ratio}")
            continue

        try:
            rect = find_bounding_box_ignoring_color(random_image01, 0)
        except Exception as e:
            continue
        if rect.width != width or rect.height != height:
            # If the object doesn't fill out the entire image, then it's not interesting.
            # My assumption is that the object fills out the entire image.
            # print(f"Avoid having an object that doesn't fill out the entire image. rect={rect}")
            continue

        flipx_image = image_flipx(random_image01)
        if np.array_equal(random_image01, flipx_image):
            continue
        flipy_image = image_flipy(random_image01)
        if np.array_equal(random_image01, flipy_image):
            continue

        color_map = {
            0: color0,
            1: color1,
        }

        result_image = image_replace_colors(random_image01, color_map)

        return result_image
    raise Exception("Failed to find a non-trivial example.")

def generate_task_mirror_swap_objects(seed: int) -> Task:
    """
    Show a objects on both sides of the mirror, in the output the objects are swapped.
    """
    the_seed = seed * 11111
    count_example = random.Random(the_seed + 1).randint(3, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    align_a = random.Random(the_seed + 3).choice(['left', 'right'])
    align_b = random.Random(the_seed + 4).choice(['left', 'right'])
    rotate_k = random.Random(the_seed + 5).randint(0, 3)

    task = Task()
    task.metadata_task_id = f'mirror a_{align_a} b_{align_b} rotate_{rotate_k}'
    min_image_size = 8
    max_image_size = MAX_IMAGE_SIZE
    half_max_width = (max_image_size - 1) // 2


    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(the_seed + 6).shuffle(colors)
    input_output_color_map = {
        0: colors[0],
        1: colors[1],
        2: colors[2],
        3: colors[3],
    }

    background_color = 0
    separator_color = 1
    object0_color0 = 2
    object0_color1 = 3

    use_same_color_for_object_and_separator = random.Random(the_seed + 6).choice([False, True])
    if use_same_color_for_object_and_separator:
        separator_color = object0_color0

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (the_seed * 37) + (retry_index * 10000) + i

            left_width = random.Random(iteration_seed + 1).randint(min_image_size, half_max_width)
            right_width = random.Random(iteration_seed + 2).randint(min_image_size, half_max_width)
            width_min = min(left_width, right_width)
            random_image_height = random.Random(iteration_seed + 4).randint(min_image_size, max_image_size)

            object0_image = random_asymmetric_object_maze(iteration_seed + 5, width_min // 2, width_min - 1, object0_color0, object0_color1)

            object0_height, object0_width = object0_image.shape

            separator_image = image_create(1, random_image_height, separator_color)
            left_image = image_create(left_width, random_image_height, background_color)
            right_image = image_create(right_width, random_image_height, background_color)

            left_input_image = left_image.copy()
            right_input_image = right_image.copy()
            left_output_image = left_image.copy()
            right_output_image = right_image.copy()

            if random_image_height < object0_height:
                # print(f"The object is too big for the image. random_image_height={random_image_height} < object0_height={object0_height}")
                continue
            position_y = random.Random(iteration_seed + 7).randint(0, random_image_height - object0_height)

            if align_a == 'left':
                left_position_x = 0
            elif align_a == 'right':
                left_position_x = left_width - object0_width
            else:
                raise Exception(f"Unknown align_a={align_a}")
            
            if align_b == 'left':
                right_position_x = 0
            elif align_b == 'right':
                right_position_x = right_width - object0_width
            else:
                raise Exception(f"Unknown align_b={align_b}")

            left_input_image = image_paste_at(object0_image, left_input_image, left_position_x, position_y)
            right_output_image = image_paste_at(object0_image, right_output_image, right_position_x, position_y)

            input_image_raw = np.hstack([left_input_image, separator_image, right_input_image])
            output_image_raw = np.hstack([left_output_image, separator_image, right_output_image])

            input_image_rotated = np.rot90(input_image_raw, k=rotate_k)
            output_image_rotated = np.rot90(output_image_raw, k=rotate_k)

            input_image = image_replace_colors(input_image_rotated, input_output_color_map)
            output_image = image_replace_colors(output_image_rotated, input_output_color_map)

            if np.array_equal(input_image, output_image):
                # No change to the image. Try again.
                continue
            break
        if output_image is None:
            raise Exception("Failed to find a non-trivial example.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_image_randomized()
    builder.append_image_rawpixel_output()
    return builder.dataset_items()

class DatasetSolveMirror(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        task = generate_task_mirror_swap_objects(seed)
        if show:
            task.show()
        transformation_id = task.metadata_task_id
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        return dataset_items

if __name__ == "__main__":
    generator = DatasetSolveMirror()
    generator.generate(
        seed=200001,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
