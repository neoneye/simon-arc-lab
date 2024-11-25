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
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_deform'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_deform.jsonl')

def generate_task_displace_rows_based_on_mask(seed: int) -> Task:
    """
    Displace rows depending on the content of the first column.
    """
    the_seed = seed * 11113
    count_example = random.Random(the_seed + 1).randint(3, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)

    task = Task()
    task.metadata_task_id = 'deform_line'
    min_image_size = 4
    max_image_size = 12

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (the_seed * 37) + (retry_index * 10000) + i
            random_image = image_create_random_advanced(iteration_seed + 1, min_image_size, max_image_size, min_image_size, max_image_size)
            random_image_height, random_image_width = random_image.shape

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

            input_image = np.hstack([mask_image, random_image])

            output_image = input_image.copy()
            for y in range(mask_height):
                if mask_image[y, 0] == 0:
                    continue
                # Displace the row.
                for x in range(random_image_width):
                    output_image[y, x + 1] = random_image[y, (x + 1) % random_image_width]

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
