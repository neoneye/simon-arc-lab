import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_shape2x2 import *
from simon_arc_lab.image_shape3x3 import *
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
    max_image_size = 4

    transformation_id = 'shape2x2'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    shape_id = random.Random(seed + 3).randint(0, 63)

    instructions = [
        f'{dataset_name} identify places where shape2x2 is {shape_id}',
        f'{dataset_name} detect shape2x2 {shape_id}',
        f'{dataset_name} find shape2x2 {shape_id}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    image_shape = ImageShape2x2.apply(input_image)

    # Places where the pixel is equal to `shape_id`, then set the pixel to 1, else set to 0.
    output_image = np.where(image_shape == shape_id, 1, 0)

    if True:
        print(f"---\ninput: {input_image}\nshape: {image_shape}\nshape_id: {shape_id}\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(image_shape, cmap='gray')
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

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=2200000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        item = generate_dataset_item_shape2x2(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    return dataset

dataset = generate_dataset(
    max_num_samples=100,
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

