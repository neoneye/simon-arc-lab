# IDEA: currently trained with image size 1-20. Next step is to train with image size 1-30.
#
# IDEA: recognize the transformation between 2 images: translate, rotate, scale, replace colors.
#
# IDEA: comparison of 2 images, are they the same, are they different
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_util import *
from simon_arc_lab.histogram import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_dataset.dataset_generator import *

SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_image_pair.jsonl')

def generate_dataset_item(seed: int) -> dict:
    """
    Two images as input.
    Do some transformation with the two images.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 20

    transformation_ids = [
        'add_histograms',
        'color_intersection',
        'number_of_intersection_colors',
        'a_remove_b_colors',
        'b_remove_a_colors',
    ]
    transformation_weights = [10, 10, 10, 10, 10]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

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
        f'{name_format}, overlap of colors of the images',
        f'Process {name_format} and return the intersection of colors',
        f'Process {name_format} and return the overlap of colors',
    ]

    instructions_number_of_intersection_colors = [
        f'{name_format}, number of colors that the images have in common',
        f'{name_format}, number of colors that the two images have in common',
        f'{name_format}, number of colors that the 2 images have in common',
        f'Process {name_format} and return the number of colors in common',
    ]

    instructions_a_remove_b_colors = [
        f'{name_format}, Remove Histogram B colors from Histogram A',
        f'{name_format}, Remove Histogram-B colors from Histogram-A',
        f'{name_format}, remove histogram-b colors from histogram-a',
        f'{name_format}, remove histogram b colors from histogram a',
        f'{name_format}, Exclude Histogram B colors from Histogram A',
        f'{name_format}, Exclude Histogram-B colors from Histogram-A',
        f'{name_format}, exclude histogram-b colors from histogram-a',
        f'{name_format}, exclude histogram b colors from histogram a',
        f'{name_format}, Histogram A without colors of Histogram B',
        f'{name_format}, Histogram-A without colors of Histogram-B',
        f'{name_format}, histogram-a without colors of histogram-b',
        f'{name_format}, histogram a without colors of histogram b',
        f'{name_format}, Histogram A excluding Histogram B colors',
        f'{name_format}, Histogram-A excluding Histogram-B colors',
        f'{name_format}, histogram-a excluding histogram-b colors',
        f'{name_format}, histogram a excluding histogram b colors',
    ]

    instructions_b_remove_a_colors = [
        f'{name_format}, Remove Histogram A colors from Histogram B',
        f'{name_format}, Remove Histogram-A colors from Histogram-B',
        f'{name_format}, remove histogram-a colors from histogram-b',
        f'{name_format}, remove histogram a colors from histogram b',
        f'{name_format}, Exclude Histogram A colors from Histogram B',
        f'{name_format}, Exclude Histogram-A colors from Histogram-B',
        f'{name_format}, exclude histogram-a colors from histogram-b',
        f'{name_format}, exclude histogram a colors from histogram b',
        f'{name_format}, Histogram B without colors of Histogram A',
        f'{name_format}, Histogram-B without colors of Histogram-a',
        f'{name_format}, histogram-b without colors of histogram-a',
        f'{name_format}, histogram b without colors of histogram a',
        f'{name_format}, Histogram B excluding Histogram A colors',
        f'{name_format}, Histogram-B excluding Histogram-a colors',
        f'{name_format}, histogram-b excluding histogram-a colors',
        f'{name_format}, histogram b excluding histogram A colors',
    ]

    instructions = None
    if transformation_id == 'add_histograms':
        instructions = instructions_add_histograms
    elif transformation_id == 'color_intersection':
        instructions = instructions_color_intersection
    elif transformation_id == 'number_of_intersection_colors':
        instructions = instructions_number_of_intersection_colors
    elif transformation_id == 'a_remove_b_colors':
        instructions = instructions_a_remove_b_colors
    elif transformation_id == 'b_remove_a_colors':
        instructions = instructions_b_remove_a_colors
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1005).choice(instructions)

    image0 = image_create_random_advanced(seed + 1006, min_image_size, max_image_size, 1, 10)
    image1 = image_create_random_advanced(seed + 1007, min_image_size, max_image_size, 1, 10)

    rle_string0 = serialize(image0)
    rle_string1 = serialize(image1)

    input = f'{rle_string0}\n{rle_string1}'

    output = None
    if transformation_id == 'add_histograms':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram0.add(histogram1).pretty()
    elif transformation_id == 'color_intersection':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram0.color_intersection_pretty(histogram1)
    elif transformation_id == 'number_of_intersection_colors':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        histogram2 = histogram0.min(histogram1)
        output = str(histogram2.number_of_unique_colors())
    elif transformation_id == 'a_remove_b_colors':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram0.remove_other_colors(histogram1).pretty()
    elif transformation_id == 'b_remove_a_colors':
        histogram0 = Histogram.create_with_image(image0)
        histogram1 = Histogram.create_with_image(image1)
        output = histogram1.remove_other_colors(histogram0).pretty()
    else:
        raise Exception("Unreachable code reached")

    max_width = max(image0.shape[1], image1.shape[1])
    max_height = max(image0.shape[0], image1.shape[0])
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_id = f'dataset=image_pair group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

class DatasetImagePair(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        item = generate_dataset_item(seed)
        return [item]

if __name__ == "__main__":
    generator = DatasetImagePair()
    generator.generate(
        seed=2100005,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()

