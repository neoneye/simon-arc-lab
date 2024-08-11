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

def generate_dataset_item_with_symmetry_output(seed: int) -> dict:
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

    image_original = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)
    image_fx = image_flipx(image_original)
    image_fy = image_flipy(image_original)
    image_180 = image_rotate_180(image_original)

    input_image = image_original.copy()

    name_image_map = [
        ('orig', image_original),
        ('flipx', image_fx),
        ('flipy', image_fy),
        ('180', image_180),
    ]
    name_image0 = random.choice(name_image_map)
    name_image1 = random.choice(name_image_map)
    name_image2 = random.choice(name_image_map)
    name_image3 = random.choice(name_image_map)
    name0, image0 = name_image0
    name1, image1 = name_image1
    name2, image2 = name_image2
    name3, image3 = name_image3

    pattern = random.choice(['hstack2', 'hstack3', 'vstack2', 'vstack3', '2x2'])
    output_image = None
    instruction_sequence = None
    if pattern == 'hstack2':
        output_image = np.hstack([image0, image1])
        instruction_sequence = f'hstack({name0} {name1})'
    if pattern == 'hstack3':
        output_image = np.hstack([image0, image1, image2])
        instruction_sequence = f'hstack({name0} {name1} {name2})'
    elif pattern == 'vstack2':
        output_image = np.vstack([image0, image1])
        instruction_sequence = f'vstack({name0} {name1})'
    elif pattern == 'vstack3':
        output_image = np.vstack([image0, image1, image2])
        instruction_sequence = f'vstack({name0} {name1}, {name2})'
    elif pattern == '2x2':
        output_image = np.vstack([np.hstack([image0, image1]), np.hstack([image2, image3])])
        instruction_sequence = f'2x2({name0} {name1} {name2} {name3})'

    assert output_image is not None
    assert instruction_sequence is not None

    instructions = [
        f'{dataset_name} symmetry {instruction_sequence}',
        f'{dataset_name} Symmetry {instruction_sequence}',
        f'{dataset_name} make symmetric {instruction_sequence}',
        f'{dataset_name} apply symmetry {instruction_sequence}',
    ]
    instruction = random.choice(instructions)

    if False:
        print(f"---\ninput: {input_image}\ninstruction: {instruction}\noutput: {output_image}")
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
    item = generate_dataset_item_with_symmetry_output(seed)
    return [item]

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1003013,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
