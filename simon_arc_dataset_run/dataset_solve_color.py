# color transformations:
# - Replace one color with another color.
# - Mask of where a particular color occurs in the input.
# - Swap 2 colors.
# - Identify the most popular color.
# - Identify the least popular color.
# - Identify both the most popular color and least popular color.
# - Swap the most/least popular colors with each other.
#
# IDEA: Image size 1xN, where N is the number of unique colors.
# IDEA: Image size 1xN, where N is the count of the most popular color.
# IDEA: Image size 1xN, where N is the count of the least popular color.
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
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_color'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_color.jsonl')

MAX_IMAGE_SIZE = 22

def generate_task_replace_color_same_palette_for_all_pairs(seed: int, transformation_id: str) -> Task:
    """
    Replace one color with another color.
    The pairs use the same palette.

    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b1948b0a
    """
    the_seed = seed * 77797

    count_example = random.Random(the_seed + 1).randint(2, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = MAX_IMAGE_SIZE
    min_padding = 0
    max_padding = 3

    uses_padding = (transformation_id == 'crop') or (transformation_id == 'padding')
    if uses_padding:
        resolved_max_image_size = max_image_size - (max_padding * 2)
    else:
        resolved_max_image_size = max_image_size

    color_padding = 0
    color_background = 1
    color_replace_from = 2
    color_replace_to = 3

    color_map_replace = {
        color_replace_from: color_replace_to,
    }

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(the_seed + 3).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    task.metadata_task_id = f'replace_color_same_palette {transformation_id}'

    for i in range(count_example+count_test):
        is_example = i < count_example

        mask_image = None
        mask_image_with_padding = None
        for retry_index in range(20):
            iteration_seed = the_seed + retry_index * 100033 + i * 1000

            width = random.Random(iteration_seed + 1).randint(min_image_size, resolved_max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, resolved_max_image_size)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, color_background, color_replace_from, ratio, iteration_seed + 4)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() != 2:
                # print(f"retry_index: {retry_index} - wrong number of colors")
                continue
            if uses_padding:
                # print(f"mask_image: {mask_image}")
                mask_image_with_padding = image_pad_random(mask_image, iteration_seed + 5, color_padding, min_padding, max_padding)
                # print(f"mask_image_with_padding: {mask_image_with_padding}")
                height2, width2 = mask_image_with_padding.shape
                if width2 > max_image_size or height2 > max_image_size:
                    # print(f"retry_index: {retry_index} - the image is too big")
                    continue

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")
        
        if uses_padding and (mask_image_with_padding is None):
            raise ValueError(f"Failed to create mask_image_with_padding")

        if transformation_id == 'no_padding':
            input_image_raw = mask_image.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'crop':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'padding':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image_with_padding.copy()
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")

        output_image_with_replaced_colors = image_replace_colors(output_image_raw, color_map_replace)
        input_image = image_replace_colors(input_image_raw, color_map)
        output_image = image_replace_colors(output_image_with_replaced_colors, color_map)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_replace_color_pairs_with_different_palettes(seed: int, transformation_id: str) -> Task:
    """
    Replace one color with another color.
    The pairs doesn't use the same palette.

    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f76d97a5
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b94a9452
    """
    the_seed = seed * 85559

    count_example = random.Random(the_seed + 1).randint(2, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = MAX_IMAGE_SIZE
    min_padding = 0
    max_padding = 3

    uses_padding = (transformation_id == 'crop') or (transformation_id == 'padding')
    if uses_padding:
        resolved_max_image_size = max_image_size - (max_padding * 2)
    else:
        resolved_max_image_size = max_image_size

    color_padding = 0
    color_background = 1
    color_replace_to = 2

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(the_seed + 3).shuffle(colors)
    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]

    available_colors = [3, 4, 5, 6, 7, 8, 9]
    random.Random(the_seed + 4).shuffle(available_colors)
    pair_colors = []
    for i in range(count_example+count_test):
        pair_color = available_colors[i % len(available_colors)]
        pair_colors.append(pair_color)

    available_palette_transformations = []
    if transformation_id == 'no_padding':
        available_palette_transformations = ['a', 'b']
    elif transformation_id == 'crop':
        available_palette_transformations = ['donothing', 'a', 'b', 'h']
    elif transformation_id == 'padding':
        available_palette_transformations = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    else:
        raise ValueError(f"Unknown transformation_id: {transformation_id}")
    palette_transformation = random.Random(the_seed + 5).choice(available_palette_transformations)

    task.metadata_task_id = f'replace_color_different_palettes {transformation_id} palette_{palette_transformation}'

    for i in range(count_example+count_test):
        is_example = i < count_example

        pair_color = pair_colors[i]

        mask_image = None
        mask_image_with_padding = None
        for retry_index in range(20):
            iteration_seed = the_seed + retry_index * 100033 + i * 1000
            width = random.Random(iteration_seed + 1).randint(min_image_size, resolved_max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, resolved_max_image_size)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, color_background, pair_color, ratio, iteration_seed + 4)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() != 2:
                # print(f"retry_index: {retry_index} - wrong number of colors")
                continue
            if uses_padding:
                # print(f"mask_image: {mask_image}")
                mask_image_with_padding = image_pad_random(mask_image, iteration_seed + 5, color_padding, min_padding, max_padding)
                # print(f"mask_image_with_padding: {mask_image_with_padding}")
                height2, width2 = mask_image_with_padding.shape
                if width2 > max_image_size or height2 > max_image_size:
                    # print(f"retry_index: {retry_index} - the image is too big")
                    continue

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")
        
        if uses_padding and (mask_image_with_padding is None):
            raise ValueError(f"Failed to create mask_image_with_padding")

        if transformation_id == 'no_padding':
            input_image_raw = mask_image.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'crop':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image.copy()
        elif transformation_id == 'padding':
            input_image_raw = mask_image_with_padding.copy()
            output_image_raw = mask_image_with_padding.copy()
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")

        # Mess with the colors of the output image
        color_map_replace_donothing = {}
        color_map_replace_a = {
            pair_color: color_replace_to,
        }
        color_map_replace_b = {
            color_background: pair_color,
            pair_color: color_replace_to,
        }
        color_map_replace_c = {
            color_padding: pair_color,
            pair_color: color_padding,
        }
        color_map_replace_d = {
            color_padding: pair_color,
            color_background: color_padding,
            pair_color: color_background,
        }
        color_map_replace_e = {
            color_background: 0,
            pair_color: 1,
        }
        color_map_replace_f = {
            color_background: 1,
            pair_color: 0,
        }
        color_map_replace_g = {
            color_background: 0,
        }
        color_map_replace_h = {
            color_background: pair_color,
            pair_color: color_background,
        }
        if palette_transformation == 'donothing':
            color_map_replace = color_map_replace_donothing
        elif palette_transformation == 'a':
            color_map_replace = color_map_replace_a
        elif palette_transformation == 'b':
            color_map_replace = color_map_replace_b
        elif palette_transformation == 'c':
            color_map_replace = color_map_replace_c
        elif palette_transformation == 'd':
            color_map_replace = color_map_replace_d
        elif palette_transformation == 'e':
            color_map_replace = color_map_replace_e
        elif palette_transformation == 'f':
            color_map_replace = color_map_replace_f
        elif palette_transformation == 'g':
            color_map_replace = color_map_replace_g
        elif palette_transformation == 'h':
            color_map_replace = color_map_replace_h
        else:
            raise ValueError(f"Unknown palette_transformation: {palette_transformation}")
        output_image_with_replaced_colors = image_replace_colors(output_image_raw, color_map_replace)

        # Assign the final colors to input/output images
        input_image = image_replace_colors(input_image_raw, color_map)
        output_image = image_replace_colors(output_image_with_replaced_colors, color_map)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_swap_colors(seed: int) -> Task:
    the_seed = seed * 55391
    count_example = random.Random(the_seed + 1).randint(2, 4)
    count_test = random.Random(the_seed + 1).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = 'swap_colors'
    min_image_size = 3
    max_image_size = MAX_IMAGE_SIZE

    for i in range(count_example+count_test):
        is_example = i < count_example

        mask_image = None
        for retry_index in range(10):
            iteration_seed = the_seed + i * 3821 + retry_index * 100033
            use_min_image_size = min_image_size
            if retry_index == 1:
                use_min_image_size = 2
            if retry_index >= 2:
                use_min_image_size = 3
            width = random.Random(iteration_seed + 1).randint(use_min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(use_min_image_size, max_image_size)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, iteration_seed + 4)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() == 2:
                # print(f"retry_index: {retry_index}")
                break

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")

        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(the_seed + 1000 + i).shuffle(colors)
        color0 = colors[0]
        color1 = colors[1]

        color_map = {
            0: color0,
            1: color1,
        }
        color_map_swapped = {
            0: color1,
            1: color0,
        }

        input_image = image_replace_colors(mask_image, color_map)
        output_image = image_replace_colors(mask_image, color_map_swapped)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_identify_most_or_least_popular_color(seed: int, find_id: str, output_size_id: str) -> Task:
    the_seed = seed * 38382351
    count_example = random.Random(the_seed + 1).randint(2, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    task.metadata_task_id = f'identify_most_or_least_popular_color {find_id} {output_size_id}'
    min_image_size = 1
    max_image_size = MAX_IMAGE_SIZE
    output_image_width = random.Random(the_seed + 3).randint(1, 5)
    output_image_height = random.Random(the_seed + 4).randint(1, 5)

    available_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(count_example+count_test):
        is_example = i < count_example

        random_image = None
        found_color = None
        number_of_retries = 0
        for retry_index in range(30):
            iteration_seed = the_seed + i * 9392 + retry_index * 100033
            use_min_image_size = min_image_size
            if retry_index == 1:
                use_min_image_size = 2
            if retry_index >= 2:
                use_min_image_size = 3
            random_image = image_create_random_advanced(iteration_seed, use_min_image_size, max_image_size, use_min_image_size, max_image_size)
            histogram = Histogram.create_with_image(random_image)
            found_color = None
            if find_id == 'most_popular':
                found_color = histogram.most_popular_color()
            elif find_id == 'least_popular':
                found_color = histogram.least_popular_color()
            else:
                raise ValueError(f"Unknown find_id: {find_id}")
            
            if found_color in available_colors:
                available_colors.remove(found_color)
            else:
                found_color = None

            if found_color is None:
                continue

            number_of_retries = retry_index
            break

        if random_image is None:
            raise ValueError(f"Failed to create random image")
        if found_color is None:
            raise ValueError(f"Failed to find color")
        if number_of_retries >= 50:
            print(f"number_of_retries: {number_of_retries}")

        input_image = random_image

        output_width = None
        output_height = None
        if output_size_id == 'NxM':
            output_width = output_image_width
            output_height = output_image_height
        elif output_size_id == 'same':
            output_width = input_image.shape[1]
            output_height = input_image.shape[0]
        else:
            raise ValueError(f"Unknown output_size_id: {output_size_id}")
        
        output_image = image_create(output_width, output_height, found_color)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_identify_most_and_least_popular_colors(seed: int) -> Task:
    the_seed = seed * 933391
    count_example = random.Random(the_seed + 1).randint(3, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    # count_test = 1

    rotate_k = random.Random(the_seed + 3).randint(0, 3)

    task = Task()
    task.metadata_task_id = f'identify_most_and_least_popular_colors rotate_{rotate_k}'
    min_image_size = 1
    max_image_size = MAX_IMAGE_SIZE

    available_colors_most_popular = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    available_colors_least_popular = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(count_example+count_test):
        is_example = i < count_example

        random_image = None
        found_color_most_popular = None
        found_color_least_popular = None
        number_of_retries = 0
        for retry_index in range(30):
            iteration_seed = the_seed + i * 9392 + retry_index * 100033
            use_min_image_size = min_image_size
            if retry_index == 1:
                use_min_image_size = 2
            if retry_index >= 2:
                use_min_image_size = 3
            random_image = image_create_random_advanced(iteration_seed, use_min_image_size, max_image_size, use_min_image_size, max_image_size)
            histogram = Histogram.create_with_image(random_image)
            if is_example and histogram.number_of_unique_colors() < 2:
                continue
            found_color_most_popular = histogram.most_popular_color()
            found_color_least_popular = histogram.least_popular_color()

            if found_color_most_popular in available_colors_most_popular and found_color_least_popular in available_colors_least_popular:
                available_colors_most_popular.remove(found_color_most_popular)
                available_colors_least_popular.remove(found_color_least_popular)
            else:
                continue

            number_of_retries = retry_index
            break

        if random_image is None:
            raise ValueError(f"Failed to create random image")
        if found_color_most_popular is None or found_color_least_popular is None:
            raise ValueError(f"Failed to find colors")
        if number_of_retries >= 50:
            print(f"number_of_retries: {number_of_retries}")

        input_image = random_image

        output_image_raw = np.array([[found_color_most_popular, found_color_least_popular]], dtype=np.uint8)
        output_image = np.rot90(output_image_raw, k=rotate_k)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_swap_most_and_least_popular_colors(seed: int) -> Task:
    the_seed = seed * 93777
    count_example = random.Random(the_seed + 1).randint(3, 4)
    count_test = random.Random(the_seed + 2).randint(1, 2)
    # count_test = 1

    task = Task()
    task.metadata_task_id = f'swap_most_and_least_popular_colors'
    min_image_size = 3
    max_image_size = MAX_IMAGE_SIZE

    available_colors_most_popular = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    available_colors_least_popular = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(count_example+count_test):
        is_example = i < count_example

        random_image = None
        found_color_most_popular = None
        found_color_least_popular = None
        number_of_retries = 0
        for retry_index in range(30):
            iteration_seed = the_seed + i * 9392 + retry_index * 100033
            use_min_image_size = min_image_size
            if retry_index == 1:
                use_min_image_size = 2
            if retry_index >= 2:
                use_min_image_size = 3
            random_image = image_create_random_advanced(iteration_seed, use_min_image_size, max_image_size, use_min_image_size, max_image_size)
            histogram = Histogram.create_with_image(random_image)
            if is_example and histogram.number_of_unique_colors() < 2:
                continue
            found_color_most_popular = histogram.most_popular_color()
            found_color_least_popular = histogram.least_popular_color()

            if found_color_most_popular in available_colors_most_popular and found_color_least_popular in available_colors_least_popular:
                available_colors_most_popular.remove(found_color_most_popular)
                available_colors_least_popular.remove(found_color_least_popular)
            else:
                continue

            number_of_retries = retry_index
            break

        if random_image is None:
            raise ValueError(f"Failed to create random image")
        if found_color_most_popular is None or found_color_least_popular is None:
            raise ValueError(f"Failed to find colors")
        if number_of_retries >= 50:
            print(f"number_of_retries: {number_of_retries}")

        input_image = random_image

        color_map_swapped = {
            found_color_most_popular: found_color_least_popular,
            found_color_least_popular: found_color_most_popular,
        }
        output_image = image_replace_colors(input_image, color_map_swapped)

        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_image_randomized()
    builder.append_image_rawpixel_output()
    return builder.dataset_items()

class DatasetSolveColor(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        j = seed % 13
        if j == 0:
            task = generate_task_replace_color_same_palette_for_all_pairs(seed, 'no_padding')
        elif j == 1:
            task = generate_task_replace_color_same_palette_for_all_pairs(seed, 'crop')
        elif j == 2:
            task = generate_task_replace_color_same_palette_for_all_pairs(seed, 'padding')
        elif j == 3:
            task = generate_task_replace_color_pairs_with_different_palettes(seed, 'no_padding')
        elif j == 4:
            task = generate_task_replace_color_pairs_with_different_palettes(seed, 'crop')
        elif j == 5:
            task = generate_task_replace_color_pairs_with_different_palettes(seed, 'padding')
        elif j == 6:
            task = generate_task_swap_colors(seed)
        elif j == 7:
            task = generate_task_identify_most_or_least_popular_color(seed, 'most_popular', 'NxM')
        elif j == 8:
            task = generate_task_identify_most_or_least_popular_color(seed, 'least_popular', 'NxM')
        elif j == 9:
            task = generate_task_identify_most_or_least_popular_color(seed, 'most_popular', 'same')
        elif j == 10:
            task = generate_task_identify_most_or_least_popular_color(seed, 'least_popular', 'same')
        elif j == 11:
            task = generate_task_identify_most_and_least_popular_colors(seed)
        elif j == 12:
            task = generate_task_swap_most_and_least_popular_colors(seed)
        else:
            raise ValueError(f"Unknown j: {j}")

        if show:        
            task.show()
        transformation_id = task.metadata_task_id
        return generate_dataset_item_list_inner(seed, task, transformation_id)

if __name__ == "__main__":
    generator = DatasetSolveColor()
    generator.generate(
        seed=230210913,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
