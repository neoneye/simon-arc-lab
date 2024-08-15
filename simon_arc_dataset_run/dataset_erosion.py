# Generate a dataset for the erosion transformation.
# A mask of what remains after 1 iteration of erosion.
#
# IDEA: Optimize. This is the slowest dataset generator that I have. It takes around 30 minutes to generate 100.000 rows.
# Had it been Rust, it would have been near instant.
#
# IDEA: Recognize the erosion mask with a particular connectivity.
#
# IDEA: Apply 2 or 3 iterations of erosion.
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
from simon_arc_lab.image_erosion import *
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'erosion'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_erosion.jsonl')

DATASET_NAMES = [
    'SIMONIMAGEEROSION',
    'SIMONSIMAGEEROSION',
    'SIMONSARCIMAGEEROSION',
    'SIMONARCIMAGEEROSION',
    'Simon-ARC-Image-Erosion',
    'Simons-ARC-Image-Erosion',
    'Simon-Image-Erosion',
    'Simons-Image-Erosion',
    'simon-arc-image-erosion',
    'simons-arc-image-erosion',
    'simon-image-erosion',
    'simons-image-erosion',
    'SimonArcImageErosion',
    'SimonsArcImageErosion',
    'SimonImageErosion',
    'SimonsImageErosion',
]

def generate_dataset_item(seed: int, connectivity: PixelConnectivity) -> dict:
    """
    Erode the image with a particular connectivity.

    :param seed: The seed for the random number generator
    :param connectivity: The pixel connectivity to use for the erosion
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 3
    max_image_size = 20

    transformation_id = 'apply_erosion'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    # get name from connectivity enum
    connectivity_name_lower = connectivity.name.lower()
    connectivity_name_upper = connectivity.name.upper()

    instructions = [
        f'{dataset_name} erosion mask {connectivity_name_lower}',
        f'{dataset_name} erosion mask {connectivity_name_upper}',
        f'{dataset_name} mask when doing erosion {connectivity_name_lower}',
        f'{dataset_name} mask when doing erosion {connectivity_name_upper}',
        f'{dataset_name} erosion mask with connectivity {connectivity_name_lower}',
        f'{dataset_name} erosion mask with connectivity {connectivity_name_upper}',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = None
    accumulated_mask = None
    has_erosion_mask = False
    for retry_index in range(100):
        seed_for_input_image = seed + 5 + retry_index * 133
        # print(f"retry_index={retry_index} seed_for_input_image={seed_for_input_image}")
        input_image = image_create_random_advanced(seed_for_input_image, min_image_size, max_image_size, min_image_size, max_image_size)
        component_list = ConnectedComponent.find_objects(connectivity, input_image)
        accumulated_mask = np.zeros_like(input_image)
        for component in component_list:
            eroded_mask = image_erosion(component, connectivity)
            accumulated_mask = np.maximum(accumulated_mask, eroded_mask)
        count = np.count_nonzero(accumulated_mask)
        if count > 0:
            has_erosion_mask = True
            break

    output_image = accumulated_mask

    if False:
        print(f"---\ninstruction: {instruction}\nconnectivity={connectivity}")
        title = connectivity_name_lower
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

    if not has_erosion_mask:
        raise Exception(f"Failed to find a image with an erosion mask. connectivity={connectivity}")

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
    seed=10101101000,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
