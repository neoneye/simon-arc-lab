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
import matplotlib.pyplot as plt

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
    benchmark_id = f'dataset=image group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

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
    benchmark_id = f'dataset=image group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

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
    benchmark_id = f'dataset=image group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=200000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        j = i % (256 + 32 + 16) 
        if j <= 16:
            item = generate_dataset_item_shape3x3_opposite(seed_start + i)
        elif j <= 32 + 16:
            item = generate_dataset_item_shape2x2(seed_start + i)
        else:
            item = generate_dataset_item_shape3x3_center(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    return dataset

dataset = generate_dataset(
    max_num_samples=100000,
    max_byte_size=1024*1024*60,
)

# Save dataset to file
filename = 'dataset_shape.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

