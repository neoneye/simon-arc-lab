# Deformation transformations.
# - displace rows depending on the content of the first column.
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
from simon_arc_lab.image_compress import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_lab.cellular_automaton import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_deform'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_deform.jsonl')

MAX_IMAGE_SIZE = 18

def generate_task_displace_rows_based_on_mask(seed: int) -> Task:
    """
    Displace rows depending on the content of the first column.
    """
    the_seed = seed * 11113
    count_example = random.Random(the_seed + 1).randint(3, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    rotate_k = random.Random(the_seed + 3).randint(0, 3)
    displacement = random.Random(the_seed + 4).choice([-1, 1])

    task = Task()
    task.metadata_task_id = f'deform_line rotate_{rotate_k} displacement_{displacement}'
    min_image_size = 4
    max_image_size = MAX_IMAGE_SIZE


    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(the_seed + 5).shuffle(colors)
    input_output_color_map = {
        0: colors[0],
        1: colors[1],
        2: colors[2],
        3: colors[3],
    }

    padding_color = random.Random(the_seed + 6).choice([0, 1])

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (the_seed * 37) + (retry_index * 10000) + i

            random_image_width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size - 2)
            random_image_height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.4, 0.5, 0.6]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            random_image_raw = image_create_random_with_two_colors(random_image_width, random_image_height, 0, 1, ratio, iteration_seed + 4)
            random_image01 = CARuleMaze().apply_wrap(random_image_raw, wrapx=False, wrapy=False, outside_value=0, step_count=3)

            color_map = {
                0: 2,
                1: 3,
            }

            random_image23 = image_replace_colors(random_image01, color_map)

            padding_right = image_create(1, random_image_height, padding_color)
            random_image_with_padding = np.hstack([random_image23, padding_right])
            random_image_with_padding_width = random_image_with_padding.shape[1]


            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 2).choice(ratios)
            mask_width = 1
            mask_height = random_image_height
            mask_image = image_create_random_with_two_colors(mask_width, mask_height, 0, 1, ratio, iteration_seed + 3)
            histogram_mask = Histogram.create_with_image(mask_image)
            if histogram_mask.number_of_unique_colors() != 2:
                # Avoid having a mask with only one color.
                continue
            if histogram_mask.get_count_for_color(0) < 2:
                # At least 2 lines of non-deformation.
                continue
            if histogram_mask.get_count_for_color(1) < 2:
                # At least 2 lines of deformation.
                continue

            input_image_raw = np.hstack([mask_image, random_image_with_padding])

            output_image_raw = input_image_raw.copy()
            for y in range(mask_height):
                if mask_image[y, 0] == 0:
                    continue
                # Displace the row.
                for x in range(random_image_with_padding_width):
                    output_image_raw[y, x + 1] = random_image_with_padding[y, (x + displacement) % random_image_with_padding_width]

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

class DatasetSolveDeform(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        task = generate_task_displace_rows_based_on_mask(seed)
        if show:
            task.show()
        transformation_id = task.metadata_task_id
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        return dataset_items

if __name__ == "__main__":
    generator = DatasetSolveDeform()
    generator.generate(
        seed=199152,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
