# Generate a dataset for the dilation transformation.
# A mask of what remains after 1 iteration of dilation.
#
# IDEA: Recognize the dilation mask with a particular connectivity.
#
# IDEA: Apply 2 or 3 iterations of dilation.
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
from simon_arc_lab.image_dilation import *
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'dilation'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_dilation.jsonl')

DATASET_NAMES = [
    'SIMONIMAGEDILATION',
    'SIMONSIMAGEDILATION',
    'SIMONSARCIMAGEDILATION',
    'SIMONARCIMAGEDILATION',
    'Simon-ARC-Image-Dilation',
    'Simons-ARC-Image-Dilation',
    'Simon-Image-Dilation',
    'Simons-Image-Dilation',
    'simon-arc-image-dilation',
    'simons-arc-image-dilation',
    'simon-image-dilation',
    'simons-image-dilation',
    'SimonArcImageDilation',
    'SimonsArcImageDilation',
    'SimonImageDilation',
    'SimonsImageDilation',
]

def generate_dataset_item(seed: int, connectivity: PixelConnectivity) -> dict:
    """
    Dilate the image with a particular connectivity.

    :param seed: The seed for the random number generator
    :param connectivity: The pixel connectivity to use for the dilation
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 3
    max_image_size = 15

    transformation_id = 'apply_dilation'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    # get name from connectivity enum
    connectivity_name_lower = connectivity.name.lower()
    connectivity_name_upper = connectivity.name.upper()

    instructions = [
        f'{dataset_name} dilation mask {connectivity_name_lower}',
        f'{dataset_name} dilation mask {connectivity_name_upper}',
        f'{dataset_name} mask when doing dilation {connectivity_name_lower}',
        f'{dataset_name} mask when doing dilation {connectivity_name_upper}',
        f'{dataset_name} dilation mask with connectivity {connectivity_name_lower}',
        f'{dataset_name} dilation mask with connectivity {connectivity_name_upper}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = None
    sum_mask = None
    has_dilation_mask = False
    for retry_index in range(100):
        seed_for_input_image = seed + 9 + retry_index * 137
        # print(f"retry_index={retry_index} seed_for_input_image={seed_for_input_image}")
        input_image = image_create_random_advanced(seed_for_input_image, min_image_size, max_image_size, min_image_size, max_image_size)
        component_list = ConnectedComponent.find_objects(connectivity, input_image)
        sum_mask = np.zeros_like(input_image)
        for component in component_list:
            dilated_mask = image_dilation(component, connectivity)
            sum_mask = np.add(sum_mask, dilated_mask)
        # The minimum value is always 1. Since a 1x1 pixel with any color, will always have a mask of 1 pixel value. Dilating that will still result in a 1 pixel value.
        # With PixelConnectivity.ALL8, the max value is 9, when there are 3x3 pixels with unique colors.
        sum_mask = np.clip(sum_mask, 1, 10)
        # Adjust the values, so they start from 0.
        sum_mask = sum_mask - 1
        count = np.count_nonzero(sum_mask)
        if count > 0:
            has_dilation_mask = True
            break

    output_image = sum_mask

    if False:
        print(f"---\ninstruction: {instruction}\nconnectivity={connectivity}")
        title = f"dilation {connectivity_name_lower}"
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

    if not has_dilation_mask:
        raise Exception(f"Failed to find a image with an dilation mask. connectivity={connectivity}")

    input = serialize(input_image)
    output = serialize(output_image)

    height, width = input_image.shape
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} connectivity={connectivity_name_lower}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    connectivity_list = [
        PixelConnectivity.NEAREST4,
        PixelConnectivity.ALL8,
        PixelConnectivity.CORNER4,
        PixelConnectivity.LR2,
        PixelConnectivity.TB2,
        PixelConnectivity.TLBR2,
        PixelConnectivity.TRBL2,
    ]
    items = []
    for index, connectivity in enumerate(connectivity_list):
        for retry_index in range(20):
            current_seed = seed + index * 10000 + 1333 * retry_index
            try:
                item = generate_dataset_item(current_seed, connectivity)
                items.append(item)
                break
            except Exception as e:
                print(f"trying again {retry_index} with connectivity {connectivity}. error: {e}")
    if len(items) == 0:
        print(f"Failed to generate any dataset items")
    return items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=12103737390,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
