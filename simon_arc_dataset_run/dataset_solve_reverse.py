# Reverse chunks of pixels in the specified direction.
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
from simon_arc_lab.image_create_random_simple import image_create_random_with_two_colors
from simon_arc_lab.cellular_automaton import *
from simon_arc_lab.histogram import *
from simon_arc_lab.task import *
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_reverse import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_reverse'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_reverse.jsonl')

def generate_task_reverse_chunks(seed: int, transformation_id: str) -> Task:
    """
    Reverse chunks of pixels in the specified direction.

    Exampe of topbottom direction:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=25d487eb
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3618c87e
    """
    count_example = random.Random(seed + 9).randint(3, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'reverse_chunks_{transformation_id}'
    min_image_size = 4
    max_image_size = 7

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 11).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    if transformation_id == 'leftright':
        direction = ReverseDirection.LEFTRIGHT
    elif transformation_id == 'topbottom':
        direction = ReverseDirection.TOPBOTTOM
    else:
        raise Exception(f"Unknown transformation_id: {transformation_id}")

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = None
        mask = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            random_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, iteration_seed + 4)

            cellularautomaton_index = random.Random(iteration_seed + 5).randint(0, 3)
            if cellularautomaton_index == 0:
                mask = CARuleGameOfLife().apply_wrap(random_image, wrapx=False, wrapy=False, outside_value=0, step_count=1)
            elif cellularautomaton_index == 1:
                mask = CARuleServiettes().apply_wrap(random_image, wrapx=False, wrapy=False, outside_value=0, step_count=1)
            elif cellularautomaton_index == 2:
                mask = CARuleCave().apply_wrap(random_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
            elif cellularautomaton_index == 3:
                mask = CARuleMaze().apply_wrap(random_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)

            # We are not interested in empty images
            histogram_mask = Histogram.create_with_image(mask)
            if histogram_mask.number_of_unique_colors() < 2:
                continue

            # We don't want almost just the background color in the input/output images
            count_mask_color1 = histogram_mask.get_count_for_color(1)
            if count_mask_color1 < 10:
                # print(f"too few pixels are going to make it to the output image, ignore this mask, count: {count_mask_color1}")
                continue

            ratios2 = [0.3, 0.4, 0.5]
            ratio2 = random.Random(iteration_seed + 6).choice(ratios2)
            random_image2 = image_create_random_with_two_colors(width, height, 1, 2, ratio2, iteration_seed + 5)

            background_image = image_create(width, height, 0)
            image_mixed = image_mix(mask, background_image, random_image2)

            # We are not interested in images with too few colors
            histogram_image = Histogram.create_with_image(image_mixed)
            if histogram_image.number_of_unique_colors() < 3:
                continue

            input_image_raw = image_mixed
            output_image_raw = image_reverse(image_mixed, 0, direction)

            # if input_image is the same as output_image, then we have to try again.
            if np.array_equal(input_image_raw, output_image_raw):
                continue

            # count the number of pixels that have the same value
            count_differences = 0
            for y in range(height):
                for x in range(width):
                    if input_image_raw[y, x] != output_image_raw[y, x]:
                        count_differences += 1

            area = width * height
            if count_differences < area * 0.08 or count_differences < 5:
                # print(f"too many static pixels, count: {count_differences}, area: {area}")
                continue

            input_image = image_replace_colors(input_image_raw, color_map)
            output_image = image_replace_colors(output_image_raw, color_map)

            break
        if (input_image is None) or (output_image is None):
            raise Exception(f"Failed to find a candidate images. seed: {seed}, transformation_id: {transformation_id} i: {i}")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_image_randomized()
    builder.append_image_rawpixel_output()
    return builder.dataset_items()

class DatasetSolveReverse(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        j = seed % 2
        if j == 0:
            task = generate_task_reverse_chunks(seed, 'leftright')
        elif j == 1:
            task = generate_task_reverse_chunks(seed, 'topbottom')
        transformation_id = task.metadata_task_id
        if show:
            task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        return dataset_items

if __name__ == "__main__":
    generator = DatasetSolveReverse()
    generator.generate(
        seed=322007771,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
