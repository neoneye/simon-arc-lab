import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_util import *
from simon_arc_lab.cellular_automaton import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced

def generate_dataset_item(seed):
    """
    Do a transformation from one image into another image.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 4
    max_image_size = 10

    transformation_ids = [
        'gameoflife_wrap',
        # 'gameoflife_nowrap',
        # 'highlife_wrap',
        # 'highlife_nowrap',
    ]
    # transformation_weights = [10, 10, 10, 10]
    transformation_weights = [10]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

    algorithm_names = [
        'SIMONCELLULARAUTOMATON',
        'SIMONCELLULARAUTOMATA',
        'SIMONSCELLULARAUTOMATON',
        'SIMONSCELLULARAUTOMATA',
        'SIMONCELLULARAUTOMATA',
        'SIMONSCELLULARAUTOMATA',
        'Simon-Cellular-Automata',
        'Simon-Cellular-Automaton',
        'Simons-Cellular-Automata',
        'Simons-Cellular-Automaton',
        'SimonCellularAutomata',
        'SimonCellularAutomaton',
        'SimonsCellularAutomata',
        'SimonsCellularAutomaton',
        'simon-cellular-automata',
        'simon-cellular-automaton',
        'simons-cellular-automata',
        'simons-cellular-automaton',
    ]
    algorithm_name = random.Random(seed + 1004).choice(algorithm_names)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    instructions_gameoflife_wrap = [
        f'{algorithm_name}, Game of Life with wrapx and wrapy. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{algorithm_name}, Game of Life with wrapxy. {color0} is dead. {color1} is alive.',
        f'{algorithm_name}, Game of Life with wrap. dead={color0} alive={color1}',
        f'{algorithm_name}, Game of Life wrap=xy. alive={color1}. dead={color0}.',
        f'{algorithm_name}, Game of Life wrap=both. live={color1}. dead={color0}.',
    ]

    instructions = None
    if transformation_id == 'gameoflife_wrap':
        instructions = instructions_gameoflife_wrap
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1005).choice(instructions)

    input_image = image_create_random_advanced(seed + 1006, min_image_size, max_image_size, min_image_size, max_image_size)
    input = serialize(input_image)

    output = None
    if transformation_id == 'gameoflife_wrap':
        dict0 = {
            color0: 0,
            color1: 1,
        }
        input_image2 = image_replace_colors(input_image, dict0)
        output_image = CARuleGameOfLife().apply_wrap(input_image2, wrapx=True, wrapy=True, outside_value=0, step_count=1)
        dict1 = {
            0: color0,
            1: color1,
        }
        output_image2 = image_replace_colors(output_image, dict1)
        output = serialize(output_image2)
    else:
        raise Exception("Unreachable code reached")

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=100000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        item = generate_dataset_item(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    return dataset

dataset = generate_dataset(
    max_num_samples=100,
    max_byte_size=1024*1024*60,
)

# Save dataset to file
filename = 'dataset_cellular_automaton.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")
