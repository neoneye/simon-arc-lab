# Symmetric transformations.
# - generate a palindrome image, hstack(original, flipx(original))
# - extract the tile that makes up the symmetric input image.
# - Use rotate cw/ccw to create a symmetric image.
# - Use flip diagonal to create a symmetric image.
# - add spacing around the input tile, so the job is to remove the spacing, and tile the image.
# - add spacing around the symmetric input image, so the job is to remove the spacing, and extract the tile image.
#
# IDEA: rotational symmetry, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2697da3f
#
# IDEA: mirror the image, where the mirrored version has a different color scheme.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6f473927
#
# IDEA: repair rotational symmetry, with red color, in order to solve the task:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1b60fb0c
#
# IDEA: currently extract a tile from a corner. Also extract the tile from the center, or another coordinate.
# IDEA: Introduce grid2x3, grid3x3, grid3x3.
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
from simon_arc_lab.image_symmetry import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_pad import image_pad_random
from simon_arc_lab.find_bounding_box import find_bounding_box_ignoring_color
from simon_arc_lab.rectangle import Rectangle
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_symmetry'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_symmetry.jsonl')

def generate_task_with_input_image_create_output_symmetry_rect(seed: int) -> Task:
    """
    Create a symmetric image from a rectangular input image.
    """
    count_example = random.Random(seed + 9).randint(2, 4)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    is_padded = random.Random(seed + 16).choice([False, True])
    max_pad_count = 8
    color_padding = random.Random(seed + 17).randint(0, 9)

    image_symmetry = ImageSymmetryRect.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)
    
    instruction_sequence = image_symmetry.instruction_sequence()
    if is_padded:
        task_id_padding_text = ' padded'
    else:
        task_id_padding_text = ''
    task.metadata_task_id = "rect " + instruction_sequence + task_id_padding_text

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_data = seed * 6 + retry_index * 137 + i
            candidate_image = image_create_random_advanced(seed_for_input_image_data, min_image_size, max_image_size, min_image_size, max_image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue
            input_image = candidate_image
            if is_padded:
                height, width = input_image.shape
                input_image = image_pad_random(input_image, seed=seed * 11 + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)
                # Reject the image if the padding color conflicts with the tile color, so it's impossible to extract the tile.
                rect = find_bounding_box_ignoring_color(input_image, color_padding)
                if rect.is_empty():
                    continue
                if rect.width != width or rect.height != height:
                    # print("ambiguous padding")
                    # The trimmed image doesn't have the same dimensions as the input_image, then there is the padding color
                    # occurs too many times in the input_image making it difficult to extract the tile.
                    continue

            output_image = image_symmetry.execute(candidate_image)
            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_input_image_create_output_symmetry_square(seed: int) -> Task:
    """
    Create a symmetric image from a square input image.

    Similar tasks:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=46442a0e
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7fe24cdd
    """
    count_example = random.Random(seed + 9).randint(3, 5)
    count_test = random.Random(seed + 10).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    is_padded = random.Random(seed + 16).choice([False, True])
    max_pad_count = 8
    color_padding = random.Random(seed + 17).randint(0, 9)

    # pattern_ids = [ImageSymmetryPatternId.HSTACK2, ImageSymmetryPatternId.VSTACK2]
    # pattern_ids = [ImageSymmetryPatternId.GRID2X2]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK3, ImageSymmetryPatternId.VSTACK3]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK4, ImageSymmetryPatternId.VSTACK4]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK5, ImageSymmetryPatternId.VSTACK5]
    # pattern_id = random.Random(seed + 773).choice(pattern_ids)
    # image_symmetry = ImageSymmetrySquare(pattern_id)
    image_symmetry = ImageSymmetrySquare.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    instruction_sequence = image_symmetry.instruction_sequence()
    if is_padded:
        task_id_padding_text = ' padded'
    else:
        task_id_padding_text = ''
    task.metadata_task_id = "square " + instruction_sequence + task_id_padding_text

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_size = seed * 5 + retry_index * 133 + i
            seed_for_input_image_data = seed * 6 + retry_index * 137 + i
            image_size = random.Random(seed_for_input_image_size).randint(min_image_size, max_image_size)
            candidate_image = image_create_random_advanced(seed_for_input_image_data, image_size, image_size, image_size, image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue
            input_image = candidate_image
            if is_padded:
                height, width = input_image.shape
                input_image = image_pad_random(input_image, seed=seed * 11 + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)
                # Reject the image if the padding color conflicts with the tile color, so it's impossible to extract the tile.
                rect = find_bounding_box_ignoring_color(input_image, color_padding)
                if rect.is_empty():
                    continue
                if rect.width != width or rect.height != height:
                    # print("ambiguous padding")
                    # The trimmed image doesn't have the same dimensions as the input_image, then there is the padding color
                    # occurs too many times in the input_image making it difficult to extract the tile.
                    continue

            output_image = image_symmetry.execute(candidate_image)
            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetry_rect_input_image_and_extract_a_particular_tile(seed: int) -> Task:
    """
    Identify the top-left rectangular tile of a symmetric image.
    """
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 4

    is_padded = random.Random(seed + 16).choice([False, True])
    max_pad_count = 8
    color_padding = random.Random(seed + 17).randint(0, 9)

    image_symmetry = ImageSymmetryRect.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    # It's the top-left tile that is always extracted. It's the first tile.
    image_symmetry.use_mutation_for_index(0, ImageSymmetryMutationId.ORIGINAL)

    instruction_sequence = image_symmetry.instruction_sequence()
    if is_padded:
        task_id_padding_text = ' padded'
    else:
        task_id_padding_text = ''
    task.metadata_task_id = "rect " + instruction_sequence + task_id_padding_text

    # Rotate the image, so the tile is not always at the top-left, but also in the other corners.
    rotate90_count = random.Random(seed + 3).randint(0, 3)
    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_size = seed * 11 + retry_index * 133 + i
            seed_for_input_image_data = seed * 17 + retry_index * 133 + i
            candidate_image = image_create_random_advanced(seed_for_input_image_data, min_image_size, max_image_size, min_image_size, max_image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue
            output_image_raw = candidate_image
            input_image_raw = image_symmetry.execute(output_image_raw)

            input_image = np.rot90(input_image_raw, rotate90_count)
            if is_padded:
                height, width = input_image.shape
                input_image = image_pad_random(input_image, seed=seed * 11 + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)
                # Reject the image if the padding color conflicts with the tile color, so it's impossible to extract the tile.
                rect = find_bounding_box_ignoring_color(input_image, color_padding)
                if rect.is_empty():
                    continue
                if rect.width != width or rect.height != height:
                    # print("ambiguous padding")
                    # The trimmed image doesn't have the same dimensions as the input_image, then there is the padding color
                    # occurs too many times in the input_image making it difficult to extract the tile.
                    continue

            output_image = np.rot90(output_image_raw, rotate90_count)
            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(seed: int) -> Task:
    """
    Identify the top-left square tile of a symmetric image.

    Similar tasks:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=73182012
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2013d3e2
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 3

    is_padded = random.Random(seed + 16).choice([False, True])
    max_pad_count = 8
    color_padding = random.Random(seed + 17).randint(0, 9)

    # pattern_ids = [ImageSymmetryPatternId.HSTACK2, ImageSymmetryPatternId.VSTACK2]
    # pattern_ids = [ImageSymmetryPatternId.GRID2X2]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK3, ImageSymmetryPatternId.VSTACK3]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK4, ImageSymmetryPatternId.VSTACK4]
    # pattern_ids = [ImageSymmetryPatternId.HSTACK5, ImageSymmetryPatternId.VSTACK5]
    # pattern_id = random.Random(seed + 773).choice(pattern_ids)
    # image_symmetry = ImageSymmetrySquare(pattern_id)
    image_symmetry = ImageSymmetrySquare.create_random(seed * 1333 + 100)
    image_symmetry.randomize_name_list(seed * 8773 + 2)

    # It's the top-left tile that is always extracted. It's the first tile.
    image_symmetry.use_mutation_for_index(0, ImageSymmetryMutationId.ORIGINAL)

    instruction_sequence = image_symmetry.instruction_sequence()
    if is_padded:
        task_id_padding_text = ' padded'
    else:
        task_id_padding_text = ''
    task.metadata_task_id = "square " + instruction_sequence + task_id_padding_text

    # Rotate the image, so the tile is not always at the top-left, but also in the other corners.
    rotate90_count = random.Random(seed + 3).randint(0, 3)
    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            seed_for_input_image_size = seed * 5 + retry_index * 133 + i
            seed_for_input_image_data = seed * 7 + retry_index * 133 + i
            image_size = random.Random(seed_for_input_image_size).randint(min_image_size, max_image_size)
            candidate_image = image_create_random_advanced(seed_for_input_image_data, image_size, image_size, image_size, image_size)
            histogram = Histogram.create_with_image(candidate_image)
            if histogram.number_of_unique_colors() < 2:
                continue
            output_image_raw = candidate_image
            input_image_raw = image_symmetry.execute(output_image_raw)

            input_image = np.rot90(input_image_raw, rotate90_count)
            if is_padded:
                height, width = input_image.shape
                input_image = image_pad_random(input_image, seed=seed * 11 + 1000 + i * 997, color=color_padding, min_pad_count=1, max_pad_count=max_pad_count)
                # Reject the image if the padding color conflicts with the tile color, so it's impossible to extract the tile.
                rect = find_bounding_box_ignoring_color(input_image, color_padding)
                if rect.is_empty():
                    continue
                if rect.width != width or rect.height != height:
                    # print("ambiguous padding")
                    # The trimmed image doesn't have the same dimensions as the input_image, then there is the padding color
                    # occurs too many times in the input_image making it difficult to extract the tile.
                    continue

            output_image = np.rot90(output_image_raw, rotate90_count)
            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_symmetry_line(seed: int) -> Task:
    """
    Symmetric pattern on both sides of the line.

    Similar tasks:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2b01abd0
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=47c1f68c
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 2
    max_image_size = 3
    min_pad_count = 0
    max_pad_count = 8
    max_wall_size = 3

    invert_variant = random.Random(seed + 3).randint(0, 7)
    is_inverted_left_input = (invert_variant & 1) > 0
    is_inverted_left_output = (invert_variant & 2) > 0
    is_inverted_right_output = (invert_variant & 4) > 0

    is_flipped = random.Random(seed + 4).choice([False, True])
    
    color_map_swap01 = {
        0: 1,
        1: 0,
    }
    color_background = 2
    color_wall = 3

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(colors)
    color_mapping = {}
    for i in range(10):
        color_mapping[i] = colors[i]

    # Swap the wall and background colors in the output image.
    swap_wall_background_colors = random.Random(seed + 6).choice([False, True])
    color_map_swap_wall_background = {
        color_background: color_wall,
        color_wall: color_background,
    }

    # What kind of rotation to use. Same for all images, or different for each image.
    rotation_n_global = random.Random(seed + 6).randint(0, 3)
    use_individual_rotation = random.Random(seed + 7).choice([False, True])
    rotate_n_list = []
    for i in range(count_example+count_test):
        if use_individual_rotation:
            rotate_n = i % 4
        else:
            rotate_n = rotation_n_global
        rotate_n_list.append(rotate_n)

    random.Random(seed + 8).shuffle(rotate_n_list)

    task.metadata_task_id = f'symmetry_line invert={invert_variant} flip={is_flipped} swapwallbackground={swap_wall_background_colors}'

    for i in range(count_example+count_test):
        rotate_n = rotate_n_list[i]
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            iteration_seed = seed * 5 + retry_index * 133 + i * 1000

            # Size of the random pattern
            random_image_width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            random_image_height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)

            # Create a two color random pattern
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(iteration_seed + 3).choice(ratios)
            random_image = image_create_random_with_two_colors(random_image_width, random_image_height, 0, 1, ratio, iteration_seed + 4)

            # Instert a tricky decoy pixel with the same color as the background or wall.
            set_x = random.Random(iteration_seed + 5).randint(0, random_image_width - 1)
            set_y = random.Random(iteration_seed + 6).randint(0, random_image_height - 1)
            set_variant = random.Random(iteration_seed + 7).randint(0, 2)
            set_color = [None, color_background, color_wall][set_variant]
            if set_color is not None:
                random_image[set_y, set_x] = set_color

            # Ensure the pattern contains both color 0 and color 1.
            histogram = Histogram.create_with_image(random_image)
            if histogram.get_count_for_color(0) == 0 or histogram.get_count_for_color(1) == 0:
                continue

            # Create the left side of the image
            left_inner = random_image.copy()

            left_inner_flipx = image_flipx(left_inner)
            if np.array_equal(left_inner, left_inner_flipx):
                continue

            # Create the right side of the image, that is a mirrored version of the left side.
            if is_flipped:
                right_inner = left_inner_flipx
            else:
                right_inner = left_inner.copy()

            # Add variable padding around the left image and right images.
            # The left/right are swapped for left/right symmetry.
            # The top/bottom padding are the same for the two images.
            seed_padding = iteration_seed + 5
            top = random.Random(seed_padding + 0).randint(min_pad_count, max_pad_count)
            bottom = random.Random(seed_padding + 1).randint(min_pad_count, max_pad_count)
            left = random.Random(seed_padding + 2).randint(min_pad_count, max_pad_count)
            distance_to_wall = random.Random(seed_padding + 3).randint(min_pad_count, max_pad_count)
            right = random.Random(seed_padding + 4).randint(min_pad_count, max_pad_count)
            left_side = np.pad(left_inner, ((top, bottom), (left, distance_to_wall)), mode='constant', constant_values=color_background)
            right_side = np.pad(right_inner, ((top, bottom), (distance_to_wall, right)), mode='constant', constant_values=color_background)

            height, right_side_width = right_side.shape

            right_side_input = image_create(right_side_width, height, color_background)

            # The wall between left and right
            wall_size = random.Random(iteration_seed + 10).randint(1, max_wall_size)
            wall = image_create(wall_size, height, color_wall)

            # Keep the pattern as it is, or invert the pattern
            left_side_input = left_side.copy()
            if is_inverted_left_input:
                left_side_input = image_replace_colors(left_side_input, color_map_swap01)
            left_side_output = left_side.copy()
            if is_inverted_left_output:
                left_side_output = image_replace_colors(left_side_output, color_map_swap01)
            right_side_output = right_side.copy()
            if is_inverted_right_output:
                right_side_output = image_replace_colors(right_side_output, color_map_swap01)

            input_image_raw = np.hstack([left_side_input, wall, right_side_input])
            output_image_raw = np.hstack([left_side_output, wall, right_side_output])

            # Swap the wall and background colors
            if swap_wall_background_colors:
                output_image_raw = image_replace_colors(output_image_raw, color_map_swap_wall_background)

            # Change palette
            input_image_raw = image_replace_colors(input_image_raw, color_mapping)
            output_image_raw = image_replace_colors(output_image_raw, color_mapping)

            # Rotate the images, so the model have to learn to detect the orientation.
            input_image = np.rot90(input_image_raw, rotate_n)
            output_image = np.rot90(output_image_raw, rotate_n)

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
        task = generate_task_with_input_image_create_output_symmetry_rect(seed)
    elif j == 1:
        task = generate_task_with_symmetry_rect_input_image_and_extract_a_particular_tile(seed)
    elif j == 2:
        task = generate_task_with_input_image_create_output_symmetry_square(seed)
    elif j == 3:
        task = generate_task_with_symmetry_square_input_image_and_extract_a_particular_tile(seed)
    elif j == 4:
        task = generate_task_with_symmetry_line(seed)
    # task.show()
    transformation_id = task.metadata_task_id
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=2849000410,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
