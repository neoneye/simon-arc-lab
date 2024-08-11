# IDEA: PixelConnectivity.CONNECTIVITYDIAGONAL4 for better understanding of diagonal shapes.
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
    max_image_size = 30

    transformation_id = 'max_mass'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    max_mass = random.Random(seed + 3).randint(1, 25)

    instructions_connectivity4 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity 4',
        f'{dataset_name} max mass {max_mass}, connectivity 4',
    ]

    instructions_connectivity8 = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity 8',
        f'{dataset_name} max mass {max_mass}, connectivity 8',
    ]

    if connectivity == PixelConnectivity.CONNECTIVITY4:
        instructions = instructions_connectivity4
    elif connectivity == PixelConnectivity.CONNECTIVITY8:
        instructions = instructions_connectivity8
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
        print(f"---\ninput: {input_image}\nmax mass: {max_mass}\nmass image: {mass_image}\noutput: {output_image}")
        plt.imshow(input_image, cmap='gray')
        plt.show()
        plt.imshow(mass_image, cmap='gray')
        plt.show()
        plt.imshow(output_image, cmap='gray')
        plt.show()

    input = serialize(input_image)
    output = serialize(output_image)

    if connectivity == PixelConnectivity.CONNECTIVITY4: 
        benchmark_connectivity = 'connectivity=4' 
    elif connectivity == PixelConnectivity.CONNECTIVITY8: 
        benchmark_connectivity = 'connectivity=8' 
    else:
        raise ValueError(f"Unknown PixelConnectivity: {connectivity}")

    width, height = input_image.shape[1], input_image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} {benchmark_connectivity} max_mass={max_mass}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    items = []
    for retry_index in range(20):
        try:
            item = generate_dataset_item_with_max_mass(seed + 10000000000 + 1333 * retry_index, PixelConnectivity.CONNECTIVITY4)
            items.append(item)
            break
        except Exception as e:
            print(f"trying again {retry_index}")
    for retry_index in range(20):
        try:
            item = generate_dataset_item_with_max_mass(seed + 20000000000 + 7711 * retry_index, PixelConnectivity.CONNECTIVITY8)
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
    seed=81905000,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
