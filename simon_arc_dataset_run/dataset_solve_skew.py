# Skew/unskew transformations.
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
from simon_arc_lab.image_skew import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_skew'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_skew.jsonl')

MAX_IMAGE_SIZE = 22

def generate_task_skew(seed: int, direction: SkewDirection) -> Task:
    """
    Skew an image.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'skew_{direction.name.lower()}'
    min_image_size = 1
    max_image_size = MAX_IMAGE_SIZE

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_map = {}
    for i, color in enumerate(colors):
        color_map[i] = color
    skew_padding_color = 0
    is_ambiguous = True
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            random_width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            random_height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            skewed_output_size = random_width + random_height - 1
            if skewed_output_size > max_image_size:
                # print("Skip too large size for output")
                continue

            random_image = image_create_random_advanced(iteration_seed + 3, random_width, random_width, random_height, random_height)

            # We are not interested in an empty image
            histogram = Histogram.create_with_image(random_image)
            if histogram.number_of_unique_colors() < 2:
                continue

            skewed_image = image_skew(random_image, skew_padding_color, direction)
            if np.array_equal(random_image, skewed_image):
                continue

            # If the the height is 1 or the width is 1, then it's ambiguous what kind of skew it is.
            input_height, input_width = random_image.shape
            if input_height > 1 and input_width > 1:
                is_ambiguous = False
            if is_ambiguous:
                # print("Skip ambiguous pair")
                continue

            output_height, output_width = skewed_image.shape

            if output_width > max_image_size or output_height > max_image_size:
                # print("Skip too large output image")
                continue

            input_image = image_replace_colors(random_image, color_map)
            output_image = image_replace_colors(skewed_image, color_map)
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    task.shuffle_examples(seed + 100)
    return task

def generate_task_unskew(seed: int, direction: SkewDirection) -> Task:
    """
    Unskew a skewed image.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'unskew_{direction.name.lower()}'
    min_image_size = 1
    max_image_size = MAX_IMAGE_SIZE

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color_map = {}
    for i, color in enumerate(colors):
        color_map[i] = color
    skew_padding_color = 0
    is_ambiguous = True
    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            random_width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            random_height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            skewed_output_size = random_width + random_height - 1
            if skewed_output_size > max_image_size:
                # print("Skip too large size for output")
                continue

            random_image = image_create_random_advanced(iteration_seed + 3, random_width, random_width, random_height, random_height)

            # We are not interested in an empty image
            histogram = Histogram.create_with_image(random_image)
            if histogram.number_of_unique_colors() < 2:
                continue

            skewed_image = image_skew(random_image, skew_padding_color, direction)
            if np.array_equal(random_image, skewed_image):
                continue

            input_height, input_width = skewed_image.shape
            if input_height > max_image_size or input_width > max_image_size:
                # print("Skip too large input image")
                continue

            # If the the height is 1 or the width is 1, then it's ambiguous what kind of skew it is.
            output_height, output_width = random_image.shape
            if output_height > 1 and output_width > 1:
                is_ambiguous = False
            if is_ambiguous:
                # print("Skip ambiguous pair")
                continue

            input_image = image_replace_colors(skewed_image, color_map)
            output_image = image_replace_colors(random_image, color_map)
            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    task.shuffle_examples(seed + 100)
    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_image_randomized()
    builder.append_image_rawpixel_output()
    return builder.dataset_items()

class DatasetSolveSkew(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        j = seed % 8
        if j == 0:
            task = generate_task_skew(seed, SkewDirection.LEFT)
        elif j == 1:
            task = generate_task_skew(seed, SkewDirection.RIGHT)
        elif j == 2:
            task = generate_task_skew(seed, SkewDirection.UP)
        elif j == 3:
            task = generate_task_skew(seed, SkewDirection.DOWN)
        elif j == 4:
            task = generate_task_unskew(seed, SkewDirection.UP)
        elif j == 5:
            task = generate_task_unskew(seed, SkewDirection.DOWN)
        elif j == 6:
            task = generate_task_unskew(seed, SkewDirection.LEFT)
        elif j == 7:
            task = generate_task_unskew(seed, SkewDirection.RIGHT)
        transformation_id = task.metadata_task_id
        if show:
            task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        return dataset_items

if __name__ == "__main__":
    generator = DatasetSolveSkew()
    generator.generate(
        seed=1223123425,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
