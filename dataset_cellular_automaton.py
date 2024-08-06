# IDEA: Exercise with bigger image. Has been exercised in the range 1-20. Sizes bigger than 20 needs to be exercised.
# IDEA: Exercise with number of steps = 2. Has most been exercises with step_count=1. A few exercises with step_count=2.
#
# IDEA: wire world
import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_util import *
from simon_arc_lab.cellular_automaton import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
import matplotlib.pyplot as plt
from dataset.dataset_generator import *

BENCHMARK_DATASET_NAME = 'cellular_automaton'
SAVE_FILENAME = 'dataset_cellular_automaton.jsonl'

DATASET_NAMES = [
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

def generate_dataset_item_transform_simple(seed):
    """
    Do a transformation from one image into another image.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 5
    max_image_size = 20

    transformation_ids = [
        'gameoflife_wrap',
        'gameoflife_nowrap',
        'highlife_wrap',
        'highlife_nowrap',
        'serviettes_wrap',
        'serviettes_nowrap',
        'cave_wrap',
        'cave_nowrap',
        'maze_wrap',
        'maze_nowrap',
    ]
    transformation_weights = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # transformation_weights = [10, 0, 10, 0, 10, 0, 10, 0, 10, 0]
    # transformation_weights = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

    dataset_name = random.Random(seed + 1004).choice(DATASET_NAMES)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]

    step_count = random.Random(seed + 1).randint(1, 1)

    instructions_gameoflife_wrap = [
        f'{dataset_name}, Game of Life with wrapx and wrapy. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Game of Life with wrapxy. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Game of Life with wrap. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Game of Life wrap=xy. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Game of Life wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, game of life wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, game-of-life wrap=both. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_gameoflife_nowrap = [
        f'{dataset_name}, Game of Life without wrap. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Game of Life with nowrap. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Game of Life with wrap=none. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Game of Life wrap=no. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Game of Life wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, game of life wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, game-of-life wrap=none. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_highlife_wrap = [
        f'{dataset_name}, HighLife with wrapx and wrapy. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, HighLife with wrapxy. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, HighLife with wrap. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, HighLife wrap=xy. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, HighLife wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, highlife wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, high-life wrap=both. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_highlife_nowrap = [
        f'{dataset_name}, HighLife without wrap. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, HighLife with nowrap. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, HighLife with wrap=none. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, HighLife wrap=no. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, HighLife wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, highlife wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, high-life wrap=none. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_serviettes_wrap = [
        f'{dataset_name}, Serviettes with wrapx and wrapy. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Serviettes with wrapxy. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Serviettes with wrap. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Serviettes wrap=xy. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Serviettes wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, serviettes wrap=both. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_serviettes_nowrap = [
        f'{dataset_name}, Serviettes without wrap. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Serviettes with nowrap. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Serviettes with wrap=none. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Serviettes wrap=no. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Serviettes wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, serviettes wrap=none. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_cave_wrap = [
        f'{dataset_name}, Cave with wrapx and wrapy. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Cave with wrapxy. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Cave with wrap. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Cave wrap=xy. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Cave wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, cave wrap=both. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_cave_nowrap = [
        f'{dataset_name}, Cave without wrap. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Cave with nowrap. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Cave with wrap=none. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Cave wrap=no. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Cave wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, cave wrap=none. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_maze_wrap = [
        f'{dataset_name}, Maze with wrapx and wrapy. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Maze with wrapxy. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Maze with wrap. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Maze wrap=xy. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Maze wrap=both. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, maze wrap=both. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions_maze_nowrap = [
        f'{dataset_name}, Maze without wrap. Steps={step_count}. Dead cells have value {color0}. Alive cells have value {color1}.',
        f'{dataset_name}, Maze with nowrap. steps={step_count}. {color0} is dead. {color1} is alive.',
        f'{dataset_name}, Maze with wrap=none. steps={step_count}. dead={color0} alive={color1}',
        f'{dataset_name}, Maze wrap=no. steps={step_count}. alive={color1}. dead={color0}.',
        f'{dataset_name}, Maze wrap=none. steps={step_count}. live={color1}. dead={color0}.',
        f'{dataset_name}, maze wrap=none. steps={step_count}. live={color1}. dead={color0}.',
    ]

    instructions = None
    if transformation_id == 'gameoflife_wrap':
        instructions = instructions_gameoflife_wrap
    elif transformation_id == 'gameoflife_nowrap':
        instructions = instructions_gameoflife_nowrap
    elif transformation_id == 'highlife_wrap':
        instructions = instructions_highlife_wrap
    elif transformation_id == 'highlife_nowrap':
        instructions = instructions_highlife_nowrap
    elif transformation_id == 'serviettes_wrap':
        instructions = instructions_serviettes_wrap
    elif transformation_id == 'serviettes_nowrap':
        instructions = instructions_serviettes_nowrap
    elif transformation_id == 'cave_wrap':
        instructions = instructions_cave_wrap
    elif transformation_id == 'cave_nowrap':
        instructions = instructions_cave_nowrap
    elif transformation_id == 'maze_wrap':
        instructions = instructions_maze_wrap
    elif transformation_id == 'maze_nowrap':
        instructions = instructions_maze_nowrap
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1005).choice(instructions)

    width = random.Random(seed + 1).randint(min_image_size, max_image_size)
    height = random.Random(seed + 2).randint(min_image_size, max_image_size)

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio = random.Random(seed + 5).choice(ratios)
    input_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, seed + 6)
    # print(input_image)

    mutate_input_id = random.Random(seed + 3).randint(0, 5)
    # print(mutate_input_id)
    if mutate_input_id == 0:
        pass
    elif mutate_input_id == 1:
        input_image = CARuleGameOfLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 2:
        input_image = CARuleHighLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 3:
        input_image = CARuleServiettes().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 4:
        input_image = CARuleCave().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 5:
        input_image = CARuleMaze().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    # print(input_image)

    output = None
    if transformation_id == 'gameoflife_wrap':
        output_image = CARuleGameOfLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
    elif transformation_id == 'gameoflife_nowrap':
        output_image = CARuleGameOfLife().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
    elif transformation_id == 'highlife_wrap':
        output_image = CARuleHighLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
    elif transformation_id == 'highlife_nowrap':
        output_image = CARuleHighLife().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
    elif transformation_id == 'serviettes_wrap':
        output_image = CARuleServiettes().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
    elif transformation_id == 'serviettes_nowrap':
        output_image = CARuleServiettes().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
    elif transformation_id == 'cave_wrap':
        output_image = CARuleCave().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
    elif transformation_id == 'cave_nowrap':
        output_image = CARuleCave().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
    elif transformation_id == 'maze_wrap':
        output_image = CARuleMaze().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
    elif transformation_id == 'maze_nowrap':
        output_image = CARuleMaze().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
    else:
        raise Exception("Unreachable code reached")
    
    color_mapping = {}
    for color_index in range(len(colors)):
        color = colors[color_index]
        color_mapping[color_index] = color

    input_image2 = image_replace_colors(input_image, color_mapping)
    output_image2 = image_replace_colors(output_image, color_mapping)
    input = serialize(input_image2)
    output = serialize(output_image2)

    # print(instruction)
    # print(input_image2)
    # print(output_image2)
    # plt.imshow(input_image, cmap='gray')
    # plt.show()
    # plt.imshow(output_image, cmap='gray')
    # plt.show()

    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} ca_step={step_count} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id,
    }
    return result_dict

def generate_dataset_item_transform_recognize(seed):
    """
    Recognize what transformation is being done from one image into another image.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 12
    max_image_size = 20

    transformation_id = 'recognize_transformation'

    dataset_name = random.Random(seed + 1).choice(DATASET_NAMES)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 2).shuffle(colors)

    step_count = random.Random(seed + 3).randint(1, 1)

    width = random.Random(seed + 4).randint(min_image_size, max_image_size)
    height = random.Random(seed + 5).randint(min_image_size, max_image_size)

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio = random.Random(seed + 6).choice(ratios)
    input_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, seed + 6)
    # print(input_image)

    mutate_input_id = random.Random(seed + 7).randint(0, 5)
    # print(mutate_input_id)
    if mutate_input_id == 0:
        pass
    elif mutate_input_id == 1:
        input_image = CARuleGameOfLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 2:
        input_image = CARuleHighLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 3:
        input_image = CARuleServiettes().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 4:
        input_image = CARuleCave().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    elif mutate_input_id == 5:
        input_image = CARuleMaze().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=1)
    # print(input_image)


    image_name_list = []
    if True:
        image = CARuleGameOfLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'gameoflife_wrap'))
    if True:
        image = CARuleGameOfLife().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'gameoflife_nowrap'))
    if True:
        image = CARuleHighLife().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'highlife_wrap'))
    if True:
        image = CARuleHighLife().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'highlife_nowrap'))
    if True:
        image = CARuleServiettes().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'serviettes_wrap'))
    if True:
        image = CARuleServiettes().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'serviettes_nowrap'))
    if True:
        image = CARuleCave().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'cave_wrap'))
    if True:
        image = CARuleCave().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'cave_nowrap'))
    if True:
        image = CARuleMaze().apply_wrap(input_image, wrapx=True, wrapy=True, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'maze_wrap'))
    if True:
        image = CARuleMaze().apply_wrap(input_image, wrapx=False, wrapy=False, outside_value=0, step_count=step_count)
        image_name_list.append((image, 'maze_nowrap'))
    
    random.Random(seed + 8).shuffle(image_name_list)

    # truncate image_name_list to a few items
    truncate_length = random.Random(seed + 9).randint(2, 5)
    image_name_list_truncated = image_name_list[:truncate_length]

    # extract list of the shuffled candidate names
    name_list = []
    for image_name_candidate in image_name_list_truncated:
        name = image_name_candidate[1]
        name_list.append(name)
    #print(name_list)
    names_with_comma = ','.join(name_list)

    instructions = [
        f'{dataset_name}, Given two images, recognize the transformations. {names_with_comma}',
        f'{dataset_name}, Given two images, recognize the transformation. {names_with_comma}',
        f'{dataset_name}, Recognize the transformation. {names_with_comma}',
        f'{dataset_name}, Recognize the transformation between input and output. {names_with_comma}',
        f'{dataset_name}, Identify the transformation. {names_with_comma}',
        f'{dataset_name}, What transformation happens. {names_with_comma}',
        f'{dataset_name}, What transformation happens here. {names_with_comma}',
    ]

    instruction = random.Random(seed + 10).choice(instructions)

    # Pick the output image
    image_name_candidate = random.Random(seed + 11).choice(image_name_list)
    output_image = image_name_candidate[0]

    color_mapping = {}
    for color_index in range(len(colors)):
        color = colors[color_index]
        color_mapping[color_index] = color

    input_image2 = image_replace_colors(input_image, color_mapping)
    output_image2 = image_replace_colors(output_image, color_mapping)
    rle_string0 = serialize(input_image2)
    rle_string1 = serialize(output_image2)

    input = f'{rle_string0}\n{rle_string1}'

    # print(instruction)
    # print(input_image2)
    # print(output_image2)
    # plt.imshow(input_image, cmap='gray')
    # plt.show()
    # plt.imshow(output_image, cmap='gray')
    # plt.show()

    # loop through the image_name_list
    # and check if there are other images that are identical to the output_image
    # if identical, then include the image_name_candidate in the image_name_list
    comparison_list = []
    for image_name in image_name_list_truncated:
        image = image_name[0]
        name = image_name[1]
        if np.array_equal(image, output_image):
            comparison_list.append(f'{name}=1')
        else:
            comparison_list.append(f'{name}=0')
    output = ','.join(comparison_list)

    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} ca_step={step_count} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id,
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = None
    if seed % 5 == 0:
        item = generate_dataset_item_transform_simple(seed)
    else:
        item = generate_dataset_item_transform_recognize(seed)
    return [item]

generator = DatasetGenerator(
    dataset_names=DATASET_NAMES,
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=4700000,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILENAME)
