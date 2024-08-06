import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_shape2x2 import *
from simon_arc_lab.image_shape3x3_center import *
from simon_arc_lab.image_shape3x3_opposite import *
from simon_arc_lab.image_shape3x3_histogram import *
import matplotlib.pyplot as plt
from dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'image'
SAVE_FILENAME = 'dataset_shape.jsonl'

DATASET_NAMES = [
    'SIMONIMAGESHAPE',
    'SIMONSIMAGESHAPE',
    'SIMONSARCIMAGESHAPE',
    'SIMONARCIMAGESHAPE',
    'Simon-ARC-Image-Shape',
    'Simons-ARC-Image-Shape',
    'Simon-Image-Shape',
    'Simons-Image-Shape',
    'simon-arc-image-shape',
    'simons-arc-image-shape',
    'simon-image-shape',
    'simons-image-shape',
    'SimonArcImageShape',
    'SimonsArcImageShape',
    'SimonImageShape',
    'SimonsImageShape',
]

def generate_dataset_item_shape2x2(seed):
    """
    Find shapes in a 2x2 neighborhood.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    transformation_id = 'shape2x2'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    shape_bit = random.Random(seed + 3).randint(0, 5)

    instructions = [
        f'{dataset_name} identify places where shape2x2 contains bit {shape_bit}',
        f'{dataset_name} detect shape2x2 bit {shape_bit}',
        f'{dataset_name} find shape2x2 bit {shape_bit}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    shape_image = ImageShape2x2.apply(input_image)

    # Places where the pixel contains the `shape_bit`, then set the pixel to 1, else set to 0.
    shape_mask = 1 << shape_bit
    output_image = np.where(shape_image & shape_mask > 0, 1, 0)

    if False:
        print(f"---\ninput: {input_image}\nshape: {shape_image}\nshape_mask: {shape_mask} (bit {shape_bit})\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(shape_image, cmap='gray')
        plt.show()
        plt.imshow(output_image, cmap='gray')
        plt.show()

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_shape3x3_center(seed):
    """
    Find shapes in a 3x3 neighborhood.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    transformation_id = 'shape3x3_center'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    shape_bit = random.Random(seed + 3).randint(0, 7)

    instructions = [
        f'{dataset_name} identify places where shape3x3 contains bit {shape_bit}',
        f'{dataset_name} detect shape3x3 bit {shape_bit}',
        f'{dataset_name} find shape3x3 bit {shape_bit}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    shape_image = ImageShape3x3Center.apply(input_image)

    # Places where the pixel contains the `shape_bit`, then set the pixel to 1, else set to 0.
    shape_mask = 1 << shape_bit
    output_image = np.where(shape_image & shape_mask > 0, 1, 0)

    if False:
        print(f"---\ninput: {input_image}\nshape: {shape_image}\nshape_mask: {shape_mask} (bit {shape_bit})\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(shape_image, cmap='gray')
        plt.show()
        plt.imshow(output_image, cmap='gray')
        plt.show()

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_shape3x3_opposite(seed):
    """
    Find shapes in a 3x3 neighborhood, where the opposite side have the same color.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    transformation_id = 'shape3x3_opposite'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    shape_bit = random.Random(seed + 3).randint(0, 3)

    instructions = [
        f'{dataset_name} identify places where shape3x3opposite contains bit {shape_bit}',
        f'{dataset_name} detect shape3x3opposite bit {shape_bit}',
        f'{dataset_name} find shape3x3opposite bit {shape_bit}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    shape_image = ImageShape3x3Opposite.apply(input_image)

    # Places where the pixel contains the `shape_bit`, then set the pixel to 1, else set to 0.
    shape_mask = 1 << shape_bit
    output_image = np.where(shape_image & shape_mask > 0, 1, 0)

    if False:
        print(f"---\ninput: {input_image}\nshape: {shape_image}\nshape_mask: {shape_mask} (bit {shape_bit})\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(shape_image, cmap='gray')
        plt.show()
        plt.imshow(output_image, cmap='gray')
        plt.show()

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_shape3x3_histogram(seed):
    """
    Find shapes in a 3x3 neighborhood, count the number of unique colors.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    transformation_ids = [
        'shape3x3_histogram_all9',
        'shape3x3_histogram_around_center',
        'shape3x3_histogram_corners',
        'shape3x3_histogram_diamond4',
        'shape3x3_histogram_diamond5',
    ]
    transformation_weights = [10, 10, 10, 10, 10]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]


    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    instructions_all9 = [
        f'{dataset_name} number of unique colors inside shape3x3histogramall9',
        f'{dataset_name} unique color count in shape3x3histogramall9',
        f'{dataset_name} shape3x3histogramall9 number of unique colors',
        f'{dataset_name} shape3x3 histogram all9 number of unique colors',
    ]

    instructions_around_center = [
        f'{dataset_name} number of unique colors inside shape3x3histogramaroundcenter',
        f'{dataset_name} unique color count in shape3x3histogramaroundcenter',
        f'{dataset_name} shape3x3histogramaroundcenter number of unique colors',
        f'{dataset_name} shape3x3 histogram around center, number of unique colors',
    ]

    instructions_corners = [
        f'{dataset_name} number of unique colors inside shape3x3histogramcorners',
        f'{dataset_name} unique color count in shape3x3histogramcorners',
        f'{dataset_name} shape3x3histogramcorners number of unique colors',
        f'{dataset_name} shape3x3 histogram corners number of unique colors',
    ]

    instructions_diamond4 = [
        f'{dataset_name} number of unique colors inside shape3x3histogramdiamond4',
        f'{dataset_name} unique color count in shape3x3histogramdiamond4',
        f'{dataset_name} shape3x3histogramdiamond4 number of unique colors',
        f'{dataset_name} shape3x3 histogram diamond4 number of unique colors',
    ]

    instructions_diamond5 = [
        f'{dataset_name} number of unique colors inside shape3x3histogramdiamond5',
        f'{dataset_name} unique color count in shape3x3histogramdiamond5',
        f'{dataset_name} shape3x3histogramdiamond5 number of unique colors',
        f'{dataset_name} shape3x3 histogram diamond5 number of unique colors',
    ]

    instructions = None
    if transformation_id == 'shape3x3_histogram_all9':
        instructions = instructions_all9
    elif transformation_id == 'shape3x3_histogram_around_center':
        instructions = instructions_around_center
    elif transformation_id == 'shape3x3_histogram_corners':
        instructions = instructions_corners
    elif transformation_id == 'shape3x3_histogram_diamond4':
        instructions = instructions_diamond4
    elif transformation_id == 'shape3x3_histogram_diamond5':
        instructions = instructions_diamond5
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    output_image = None
    if transformation_id == 'shape3x3_histogram_all9':
        output_image = ImageShape3x3Histogram.number_of_unique_colors_all9(input_image)
    elif transformation_id == 'shape3x3_histogram_around_center':
        output_image = ImageShape3x3Histogram.number_of_unique_colors_around_center(input_image)
    elif transformation_id == 'shape3x3_histogram_corners':
        output_image = ImageShape3x3Histogram.number_of_unique_colors_in_corners(input_image)
    elif transformation_id == 'shape3x3_histogram_diamond4':
        output_image = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(input_image)
    elif transformation_id == 'shape3x3_histogram_diamond5':
        output_image = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(input_image)
    else:
        raise Exception("Unreachable code reached")

    if False:
        print(f"---\ninstruction: {instruction}\ninput: {input_image}\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(output_image, cmap='gray')
        plt.show()

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = None
    j = seed % 4
    if j == 0:
        item = generate_dataset_item_shape3x3_opposite(seed)
    elif j == 1:
        item = generate_dataset_item_shape2x2(seed)
    elif j == 2:
        item = generate_dataset_item_shape3x3_center(seed)
    else:
        item = generate_dataset_item_shape3x3_histogram(seed)
    return [item]

generator = DatasetGenerator(
    dataset_names=DATASET_NAMES,
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=5800057,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILENAME)
