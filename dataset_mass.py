import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
import matplotlib.pyplot as plt
from dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'image'
SAVE_FILENAME = 'dataset_mass.jsonl'

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

def generate_dataset_item_with_max_mass(seed):
    """
    Find objects less than or equal to a particular mass.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 10

    transformation_id = 'max_mass'

    dataset_name = random.Random(seed + 2).choice(DATASET_NAMES)

    max_mass = random.Random(seed + 3).randint(1, 4)

    instructions = [
        f'{dataset_name} identify places where max mass is {max_mass}, connectivity 4',
        f'{dataset_name} max mass {max_mass}, connectivity 4',
    ]

    instruction = random.Random(seed + 4).choice(instructions)

    input_image = image_create_random_advanced(seed + 5, min_image_size, max_image_size, min_image_size, max_image_size)

    ignore_mask = np.zeros_like(input_image)
    component_list = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.CONNECTIVITY4, input_image, ignore_mask)
    # print(f"component_list: {component_list}")

    width = input_image.shape[1]
    height = input_image.shape[0]

    mask = np.zeros_like(input_image)
    for component in component_list:
        if component.mass > max_mass:
            continue
        for y in range(height):
            for x in range(width):
                if component.mask[y, x] == 1:
                    mask[y, x] = 1

    output_image = mask

    if True:
        print(f"---\ninput: {input_image}\nmax mass: {max_mass}\noutput: {output_image}")
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
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = generate_dataset_item_with_max_mass(seed)
    return [item]

generator = DatasetGenerator(
    dataset_names=DATASET_NAMES,
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=4900000,
    max_num_samples=5,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILENAME)
