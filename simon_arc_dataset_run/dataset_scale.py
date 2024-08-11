# Model trained with max_scale_factor: 1-3. Exercise the model with a bigger max_scale_factor.
#
# IDEA: Add noise when doing down scaling, so the LLM learns to ignore the noise.
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
from simon_arc_lab.show_prediction_result import show_prediction_result
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

def generate_dataset_item_with_scale(seed: int) -> dict:
    """
    Resize the image by a scale factor.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    min_scale_factor = 1
    max_scale_factor = 3

    transformation_id = 'scale'

    random.seed(seed)

    dataset_name = random.choice(DATASET_NAMES)

    scale_up_down = ['up', 'down']
    scalex_up_down = random.choice(scale_up_down)
    scaley_up_down = random.choice(scale_up_down)

    scalex_factor = 1
    scaley_factor = 1
    for _ in range(100):
        scalex_factor = random.randint(min_scale_factor, max_scale_factor)
        scaley_factor = random.randint(min_scale_factor, max_scale_factor)
        if scalex_factor == 1 and scaley_factor == 1:
            continue
        break
    if scalex_factor == 1 and scaley_factor == 1:
        if seed % 2 == 0:
            scalex_factor = 1
            scaley_factor = 2
        else:
            scalex_factor = 2
            scaley_factor = 1

    instructions_scale_both_x_and_y = [
        f'{dataset_name} scale-x {scalex_up_down} by {scalex_factor}, scale-y {scaley_up_down} by {scaley_factor}',
        f'{dataset_name} scalex={scalex_up_down}{scalex_factor} scaley={scaley_up_down}{scaley_factor}',
    ]
    instructions_scale_x_with_y1 = [
        f'{dataset_name} scale-x {scalex_up_down} by {scalex_factor}',
        f'{dataset_name} scalex={scalex_up_down}{scalex_factor}',
    ]
    instructions_scale_y_with_x1 = [
        f'{dataset_name} scale-y {scaley_up_down} by {scaley_factor}',
        f'{dataset_name} scaley={scaley_up_down}{scaley_factor}',
    ]

    instructions = instructions_scale_both_x_and_y
    if scalex_factor == 1:
        instructions = instructions_scale_y_with_x1
    if scaley_factor == 1:
        instructions = instructions_scale_x_with_y1
    instruction = random.choice(instructions)

    unscaled_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    # Input image
    input_image = unscaled_image.copy()
    if scalex_up_down == 'down':
        input_image = np.kron(input_image, np.ones((1, scalex_factor))).astype(np.uint8)
    if scaley_up_down == 'down':
        input_image = np.kron(input_image, np.ones((scaley_factor, 1))).astype(np.uint8)

    # Output image
    output_image = unscaled_image.copy()
    if scalex_up_down == 'up':
        output_image = np.kron(output_image, np.ones((1, scalex_factor))).astype(np.uint8)
    if scaley_up_down == 'up':
        output_image = np.kron(output_image, np.ones((scaley_factor, 1))).astype(np.uint8)

    if False:
        print(f"---\ninput: {input_image}\nscale-x {scalex_up_down} by {scalex_factor}\nscale-y {scaley_up_down} by {scaley_factor}\noutput: {output_image}")
        # title = f'scale-x {scalex_up_down} by {scalex_factor}, scale-y {scaley_up_down} by {scaley_factor}'
        title = instruction
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_params = f'scalex={scalex_up_down}{scalex_factor} scaley={scaley_up_down}{scaley_factor}'
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} {benchmark_params}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = generate_dataset_item_with_scale(seed)
    return [item]

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=3003003,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
