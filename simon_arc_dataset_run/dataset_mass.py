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

BENCHMARK_DATASET_NAME = 'mass'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_mass.jsonl')

DATASET_NAMES = [
    'SIMONIMAGEMASS',
    'SIMONSIMAGEMASS',
    'SIMONSARCIMAGEMASS',
    'SIMONARCIMAGEMASS',
    'Simon-ARC-Image-Mass',
    'Simons-ARC-Image-Mass',
    'Simon-Image-Mass',
    'Simons-Image-Mass',
    'simon-arc-image-mass',
    'simons-arc-image-mass',
    'simon-image-mass',
    'simons-image-mass',
    'SimonArcImageMass',
    'SimonsArcImageMass',
    'SimonImageMass',
    'SimonsImageMass',
]

def generate_dataset_item_with_max_mass(seed: int, connectivity: PixelConnectivity) -> dict:
    """
    Find objects less than or equal to a particular mass.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 20

    transformation_id = 'max_mass'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    max_mass = random.Random(seed + 3).randint(1, 8)

    instructions_connectivity_nearest4 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity 4',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity nearest4',
        f'{dataset_name} max mass {max_mass}, connectivity 4',
        f'{dataset_name} max mass {max_mass}, connectivity nearest4',
    ]

    instructions_connectivity_all8 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity 8',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity all8',
        f'{dataset_name} max mass {max_mass}, connectivity 8',
        f'{dataset_name} max mass {max_mass}, connectivity all8',
    ]

    instructions_connectivity_corner4 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity corner4',
        f'{dataset_name} max mass {max_mass}, connectivity corner4',
    ]

    instructions_connectivity_lr2 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity lr2',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity leftright2',
        f'{dataset_name} max mass {max_mass}, connectivity lr2',
        f'{dataset_name} max mass {max_mass}, connectivity leftright2',
    ]

    instructions_connectivity_tb2 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity tb2',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity topbottom2',
        f'{dataset_name} max mass {max_mass}, connectivity tb2',
        f'{dataset_name} max mass {max_mass}, connectivity topbottom2',
    ]

    instructions_connectivity_tlbr2 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity tlbr2',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity topleftbottomright2',
        f'{dataset_name} max mass {max_mass}, connectivity tlbr2',
        f'{dataset_name} max mass {max_mass}, connectivity topleftbottomright2',
    ]

    instructions_connectivity_trbl2 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity trbl2',
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity toprightbottomleft2',
        f'{dataset_name} max mass {max_mass}, connectivity trbl2',
        f'{dataset_name} max mass {max_mass}, connectivity toprightbottomleft2',
    ]

    if connectivity == PixelConnectivity.NEAREST4:
        instructions = instructions_connectivity_nearest4
    elif connectivity == PixelConnectivity.ALL8:
        instructions = instructions_connectivity_all8
    elif connectivity == PixelConnectivity.CORNER4:
        instructions = instructions_connectivity_corner4
    elif connectivity == PixelConnectivity.LR2:
        instructions = instructions_connectivity_lr2
    elif connectivity == PixelConnectivity.TB2:
        instructions = instructions_connectivity_tb2
    elif connectivity == PixelConnectivity.TLBR2:
        instructions = instructions_connectivity_tlbr2
    elif connectivity == PixelConnectivity.TRBL2:
        instructions = instructions_connectivity_trbl2
    else:
        raise ValueError(f"Unknown PixelConnectivity: {connectivity}")

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    component_list = ConnectedComponent.find_objects(connectivity, input_image)
    # print(f"component_list: {component_list}")
    if len(component_list) == 0:
        mass_image = np.zeros_like(input_image)
    else:
        mass_image = object_mass(component_list)

    width = input_image.shape[1]
    height = input_image.shape[0]

    mask = np.zeros_like(input_image)
    for y in range(height):
        for x in range(width):
            mass = mass_image[y, x]
            if mass == 0 or mass > max_mass:
                continue
            mask[y, x] = 1

    output_image = mask

    if False:
        # print(f"---\ninput: {input_image}\nmax mass: {max_mass}\nmass image: {mass_image}\noutput: {output_image}")
        print(f"---\ninstruction: {instruction}\nmax mass: {max_mass}\nconnectivity={connectivity}")
        title = f"{connectivity} max_mass={max_mass}"
        show_prediction_result(input_image, mass_image, output_image, title, show_grid=True, save_path=None)

    input = serialize(input_image)
    output = serialize(output_image)

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_connectivity = f'connectivity={connectivity}' 
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} {benchmark_connectivity} max_mass={max_mass}'

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
                item = generate_dataset_item_with_max_mass(seed + index * 10000 + 1333 * retry_index, connectivity)
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
