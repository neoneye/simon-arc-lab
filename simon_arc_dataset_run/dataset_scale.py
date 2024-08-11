import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_object_mass import *
import matplotlib.pyplot as plt
from simon_arc_dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'scale'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_scale.jsonl')

DATASET_NAMES = [
    'SIMONIMAGESCALE',
    'SIMONSIMAGESCALE',
    'SIMONSARCIMAGESCALE',
    'SIMONARCIMAGESCALE',
    'Simon-ARC-Image-Scale',
    'Simons-ARC-Image-Scale',
    'Simon-Image-Scale',
    'Simons-Image-Scale',
    'simon-arc-image-scale',
    'simons-arc-image-scale',
    'simon-image-scale',
    'simons-image-scale',
    'SimonArcImageScale',
    'SimonsArcImageScale',
    'SimonImageScale',
    'SimonsImageScale',
]

def generate_dataset_item_with_scaleup(seed: int) -> dict:
    """
    Resize the image by a scale factor.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 5

    transformation_id = 'scaleup'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    scaleup_factor = random.Random(seed + 3).randint(2, 4)

    instructions = [
        f'{dataset_name} scaleup by {scaleup_factor}',
        f'{dataset_name} scale up by {scaleup_factor}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)
    output_image = np.kron(input_image, np.ones((scaleup_factor, scaleup_factor)))

    if True:
        print(f"---\ninput: {input_image}\nscaleup_factor: {scaleup_factor}\n\noutput: {output_image}")
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
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} scaleup_factor={scaleup_factor}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = generate_dataset_item_with_scaleup(seed)
    return [item]

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1003003,
    max_num_samples=1,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
