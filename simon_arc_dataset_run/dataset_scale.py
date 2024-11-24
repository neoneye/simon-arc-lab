# image scaling transformations:
# - scale the input image by a scale factor, both in the x and y directions, and do up/down scaling.
# - recognize the transformation between two images, identifying what kind of scaling happened.
#
# Model trained with max_scale_factor: 1-5. Exercise the model with a bigger max_scale_factor.
#
# IDEA: Add noise when doing down scaling, so the LLM learns to ignore the noise.
#
# IDEA: When recognizing the transformation, show garbage data in the image, and have the LLM ignore the garbage data.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_lab.image_scale import image_scale
from simon_arc_dataset.dataset_generator import DatasetGenerator

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

def compute_max_image_size(max_image_size: int, scale_factor: int) -> int:
    computed_max_image_size = max_image_size
    if scale_factor >= 2:
        computed_max_image_size = max_image_size // scale_factor
        if computed_max_image_size < 1:
            computed_max_image_size = 1

    # print(f"scale_factor: {scale_factor} computed_max_image_size {computed_max_image_size}")
    return computed_max_image_size


def generate_dataset_item_with_scale(seed: int, show: bool) -> dict:
    """
    Resize the image by a scale factor.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    min_scale_factor = 1
    max_scale_factor = 7
        
    transformation_id = 'scale_input'

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

    computed_x_max_image_size = compute_max_image_size(max_image_size, scalex_factor)
    computed_y_max_image_size = compute_max_image_size(max_image_size, scaley_factor)

    unscaled_image = image_create_random_advanced(seed + 5, min_image_size, computed_x_max_image_size, min_image_size, computed_y_max_image_size)
    input_image, output_image = image_scale(unscaled_image, scalex_up_down, scalex_factor, scaley_up_down, scaley_factor)

    if show:
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

def format_scalexy_identifier(x_up_down: str, x_scale: int, y_up_down: str, y_scale: int) -> str:
    """
    the x_up_down and y_up_down are either 'up' or 'down'
    the x_scale and y_scale are positive integers in the range 1 to max_scale_factor
    """
    if x_scale == 1:
        x_suffix = ''
    else:
        x_suffix = str(x_up_down)
    if y_scale == 1:
        y_suffix = ''
    else:
        y_suffix = str(y_up_down)
    return f'x{x_scale}{x_suffix}_y{y_scale}{y_suffix}'

def format_scalexy_identifiers(max_scale_factor: int) -> list[str]:
    name_list = []
    up_down = ['up', 'down']
    for x_up_down in up_down:
        for y_up_down in up_down:
            for y_scale in range(1, max_scale_factor + 1):
                for x_scale in range(1, max_scale_factor + 1):
                    name = format_scalexy_identifier(x_up_down, x_scale, y_up_down, y_scale)
                    name_list.append(name)
    return name_list

def generate_dataset_item_transform_recognize(seed: int, show: bool) -> dict:
    """
    Recognize what transformation is being done from one image into another image.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    min_scale_factor = 1
    max_scale_factor = 7

    transformation_id = 'recognize_transformation'

    random.seed(seed)

    dataset_name = random.choice(DATASET_NAMES)

    scale_up_down = ['up', 'down']
    scalex_up_down = random.choice(scale_up_down)
    scaley_up_down = random.choice(scale_up_down)

    scalex_factor = random.randint(min_scale_factor, max_scale_factor)
    scaley_factor = random.randint(min_scale_factor, max_scale_factor)

    current_name_id = format_scalexy_identifier(scalex_up_down, scalex_factor, scaley_up_down, scaley_factor)
    name_list = format_scalexy_identifiers(max_scale_factor)
    name_list.remove(current_name_id)

    # print(f"current_name_id: {current_name_id}")
    # print(name_list)

    # create a shuffled list of names, where the current transformation is included half of the time
    # truncate image_name_list to a few items
    truncate_length = random.randint(2, 5)
    name_list_truncated = name_list[:truncate_length]

    # Half of the time, include the current transformation
    if random.randint(0, 1) == 1:
        name_list_truncated[0] = current_name_id

    random.shuffle(name_list_truncated)
    # print(f"name_list_truncated: {name_list_truncated}")
    
    names_with_comma = ','.join(name_list_truncated)
    # print(f"names_with_comma: {names_with_comma}")

    instructions = [
        f'{dataset_name}, Given two images, recognize the transformations. {names_with_comma}',
        f'{dataset_name}, Given two images, recognize the transformation. {names_with_comma}',
        f'{dataset_name}, Recognize the transformation. {names_with_comma}',
        f'{dataset_name}, Recognize the transformation between input and output. {names_with_comma}',
        f'{dataset_name}, Identify the transformation. {names_with_comma}',
        f'{dataset_name}, What transformation happens. {names_with_comma}',
        f'{dataset_name}, What transformation happens here. {names_with_comma}',
    ]
    instruction = random.choice(instructions)

    computed_x_max_image_size = compute_max_image_size(max_image_size, scalex_factor)
    computed_y_max_image_size = compute_max_image_size(max_image_size, scaley_factor)

    unscaled_image = image_create_random_advanced(seed + 5, min_image_size, computed_x_max_image_size, min_image_size, computed_y_max_image_size)
    input_image, output_image = image_scale(unscaled_image, scalex_up_down, scalex_factor, scaley_up_down, scaley_factor)

    input_rle = serialize(input_image)
    output_rle = serialize(output_image)
    input = f'{input_rle}\n{output_rle}'

    items = []
    for name in name_list_truncated:
        if name == current_name_id:
            value = 1
        else:
            value = 0
        items.append(f'{name}={value}')
    output = ','.join(items)

    if show:
        print(f"instruction: {instruction}\noutput: {output}")
        print(f"---\ninput: {input_image}\nscale-x {scalex_up_down} by {scalex_factor}\nscale-y {scaley_up_down} by {scaley_factor}\noutput: {output_image}")
        # title = f'scale-x {scalex_up_down} by {scalex_factor}, scale-y {scaley_up_down} by {scaley_factor}'
        title = instruction
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

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

class DatasetScale(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        if seed % 2 == 0:
            item = generate_dataset_item_with_scale(seed, show)
        else:
            item = generate_dataset_item_transform_recognize(seed, show)
        return [item]

if __name__ == "__main__":
    generator = DatasetScale()
    generator.generate(
        seed=10003005,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
