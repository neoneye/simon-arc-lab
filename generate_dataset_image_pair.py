import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_util import *
from simon_arc_lab.histogram import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced

def generate_dataset_item(seed):
    """
    Two images as input.
    Do some transformation with the two images.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 10

    instruction_ids = [
        'add_histograms',
        'color_intersection',
        'number_of_intersection_colors',
    ]
    instruction_weights = [10, 10, 10]
    instruction_id = random.Random(seed + 1001).choices(instruction_ids, weights=instruction_weights, k=1)[0]

    name_formats = [
        'SIMONARCRLEIMAGEPAIR',
        'SIMONSARCRLEIMAGEPAIR',
        'SIMONSARCIMAGEPAIR',
        'SIMONARCIMAGEPAIR',
        'Simon-ARC-RLE-Image-Pair',
        'Simons-ARC-RLE-Image-Pair',
        'Simons-ARC-Image-Pair',
        'Simons-Arc-Image-Pair',
        'SimonsArcRleImagePair',
        'SimonsRLEImagePair',
        'SimonRLEImagePair',
        'simons-rle-image-pair',
        'simons-image-pair',
        'simons-arc-image-pair',
        'simons-arc-rle-image-pair',
        'simon-rle-image-pair',
        'simon-image-pair',
        'simon-arc-image-pair',
        'simon-arc-rle-image-pair',
    ]
    name_format = random.Random(seed + 1004).choice(name_formats)

    instructions_add_histograms = [
        f'Add histograms of two images of type {name_format}',
        f'Two images of type {name_format}, add histograms',
        f'Here are 2 images of type {name_format}, add histograms',
        f'{name_format}, add histograms',
        f'{name_format}, sum histograms',
        f'Process {name_format} and return the histogram sum',
        f'Process {name_format} and return the sum of histograms',
    ]

    instructions_color_intersection = [
        f'{name_format}, unique colors that the images have in common',
        f'{name_format}, unique colors that the two images have in common',
        f'{name_format}, unique colors that the 2 images have in common',
        f'{name_format}, intersection of colors of the two images',
        f'{name_format}, intersection of colors of the 2 images',
        f'{name_format}, intersection of colors of the images',
        f'Process {name_format} and return the intersection of colors',
    ]

    instructions_number_of_intersection_colors = [
        f'{name_format}, number of colors that the images have in common',
        f'{name_format}, number of colors that the two images have in common',
        f'{name_format}, number of colors that the 2 images have in common',
        f'Process {name_format} and return the number of colors in common',
    ]

    instructions = None
    if instruction_id == 'add_histograms':
        instructions = instructions_add_histograms
    elif instruction_id == 'color_intersection':
        instructions = instructions_color_intersection
    elif instruction_id == 'number_of_intersection_colors':
        instructions = instructions_number_of_intersection_colors
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1005).choice(instructions)

    image0 = image_create_random_advanced(seed + 1006, min_image_size, max_image_size, min_image_size, max_image_size)
    image1 = image_create_random_advanced(seed + 1007, min_image_size, max_image_size, min_image_size, max_image_size)

    rle_string0 = serialize(image0)
    rle_string1 = serialize(image1)

    input = f'{rle_string0}\n{rle_string1}'

    output = None
    if instruction_id == 'add_histograms':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram0.add(histogram1).pretty()
    elif instruction_id == 'color_intersection':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram0.color_intersection_pretty(histogram1)
    elif instruction_id == 'number_of_intersection_colors':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        histogram2 = histogram0.min(histogram1)
        output = str(histogram2.number_of_unique_colors())
    else:
        raise Exception("Unreachable code reached")

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=500000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        item = generate_dataset_item(seed_start + i)
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
filename = 'dataset_image_pair.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

