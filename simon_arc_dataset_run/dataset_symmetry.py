import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.image_util import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_object_mass import *
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'symmetry'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_symmetry.jsonl')

DATASET_NAMES = [
    'SIMONIMAGESYMMETRY',
    'SIMONSIMAGESYMMETRY',
    'SIMONSARCIMAGESYMMETRY',
    'SIMONARCIMAGESYMMETRY',
    'Simon-ARC-Image-Symmetry',
    'Simons-ARC-Image-Symmetry',
    'Simon-Image-Symmetry',
    'Simons-Image-Symmetry',
    'simon-arc-image-symmetry',
    'simons-arc-image-symmetry',
    'simon-image-symmetry',
    'simons-image-symmetry',
    'SimonArcImageSymmetry',
    'SimonsArcImageSymmetry',
    'SimonImageSymmetry',
    'SimonsImageSymmetry',
]

def generate_dataset_item_with_symmetry(seed: int) -> dict:
    """
    Make a symmetric image.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 10

    transformation_id = 'symmetry'

    random.seed(seed)

    dataset_name = random.choice(DATASET_NAMES)

    instructions = [
        f'{dataset_name} symmetry',
    ]
    instruction = random.choice(instructions)

    image1 = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)
    image2 = image_flipx(image1)
    image3 = image_flipy(image1)
    image4 = image_rotate_180(image1)

    input_image = image1.copy()

    pattern = random.choice(['horz', 'vert', '2x2'])
    if pattern == 'horz':
        output_image = np.hstack([image1, image3])
    elif pattern == 'vert':
        output_image = np.vstack([image1, image2])
    elif pattern == '2x2':
        output_image = np.vstack([np.hstack([image1, image3]), np.hstack([image2, image4])])

    if True:
        print(f"---\ninput: {input_image}\n\noutput: {output_image}")
        title = instruction
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

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
    item = generate_dataset_item_with_symmetry(seed)
    return [item]

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1003011,
    max_num_samples=3,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
