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

def generate_dataset_item(seed: int, xconnectivity: PixelConnectivity) -> dict:
    """
    Find objects less than or equal to a particular mass.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 5
    max_image_size = 10

    transformation_id = 'max_mass'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    connectivity = PixelConnectivity.ALL8
    erosion_id = ImageErosionId.ALL8
    erosion_name = 'all8'

    instructions_all8 = [
        f'{dataset_name} erosion mask all8',
        f'{dataset_name} mask when doing erosion ALL8',
        f'{dataset_name} erosion mask with connectivity all8',
    ]
    instructions = instructions_all8

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = None
    accumulated_mask = None
    found = False
    for retry_index in range(100):
        input_image = image_create_random_advanced(seed + 5 + retry_index * 133, min_image_size, max_image_size, min_image_size, max_image_size)
        component_list = ConnectedComponent.find_objects(connectivity, input_image)
        accumulated_mask = np.zeros_like(input_image)
        for component in component_list:
            eroded_mask = image_erosion(component, erosion_id)
            accumulated_mask = np.maximum(accumulated_mask, eroded_mask)
        count = np.count_nonzero(accumulated_mask)
        if count > 0:
            found = True
            break
    if not found:
        raise Exception("Failed to find a image with an erosion mask.")

    output_image = accumulated_mask

    if False:
        print(f"---\ninstruction: {instruction}\nerosion_id={erosion_id}")
        title = f"{erosion_name}"
        show_prediction_result(input_image, output_image, None, title, show_grid=True, save_path=None)

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} erosion_id={erosion_id}'

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
            try:
                item = generate_dataset_item(seed + index * 10000 + 1333 * retry_index, connectivity)
                items.append(item)
                break
            except Exception as e:
                print(f"trying again {retry_index}")
    if len(items) == 0:
        print(f"Failed to generate any dataset items")
    return items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1401905000,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
