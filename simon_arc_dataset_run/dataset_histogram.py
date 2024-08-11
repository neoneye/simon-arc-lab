# IDEA: comparision of 3 or more histograms and extract: intersection, union.
# IDEA: Exercise much bigger values. Currently have been exercising the range 1-1600.
#
# IDEA: What are the most popular colors in the histogram?
# IDEA: What are the least popular colors in the histogram?
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import random
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_dataset.dataset_generator import *

BENCHMARK_DATASET_NAME_ONE = 'histogram_one'
BENCHMARK_DATASET_NAME_TWO = 'histogram_two'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_histogram.jsonl')

DATASET_NAMES = [
    'SIMONARCHISTOGRAM',
    'SIMON-ARC-HISTOGRAM',
    'SIMONS-ARC-HISTOGRAM',
    'SIMONS-HISTOGRAM',
    'Simon-ARC-Histogram',
    'SimonsHistogram',
    'Simons-Histogram',
    'simons-Arc-Histogram',
    'simon-histogram',
    'simonhistogram',
    'simonshistogram',
]

def generate_one_histogram_dataset_item(seed):
    transformation_ids = [
        'number_of_unique_colors',
        'unique_colors',
    ]
    transformation_weights = [20, 20]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

    dataset_name = random.Random(seed + 1005).choice(DATASET_NAMES)

    instructions_number_of_unique_colors = [
        f'{dataset_name}, number of unique colors',
        f'number of unique colors in {dataset_name}',
        f'how many unique colors are there in {dataset_name}',
        f'{dataset_name}, unique color count',
    ]

    instructions_unique_colors = [
        f'{dataset_name}, unique colors',
        f'unique colors in {dataset_name}',
        f'unique colors of {dataset_name}',
        f'what are the unique colors in {dataset_name}',
        f'{dataset_name}, unique color list',
        f'{dataset_name}, unique colors',
    ]

    instructions = None
    if transformation_id == 'number_of_unique_colors':
        instructions = instructions_number_of_unique_colors
    elif transformation_id == 'unique_colors':
        instructions = instructions_unique_colors
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1006).choice(instructions)

    max_color_value = 6400
    histogram = Histogram.create_random(seed + 1007, 1, 9, 1, max_color_value)

    input = histogram.pretty()

    output = None
    if transformation_id == 'number_of_unique_colors':
        output = str(histogram.number_of_unique_colors())
    elif transformation_id == 'unique_colors':
        output = histogram.unique_colors_pretty()
    else:
        raise Exception("Unreachable code reached")

    sum_of_counters = histogram.sum_of_counters()
    benchmark_histogram_size = histogram_total_to_string(sum_of_counters)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME_ONE
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} histogram_size={benchmark_histogram_size}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id,
    }
    return result_dict

def generate_two_histogram_dataset_item(seed):
    transformation_ids = [
        'add', 
        'subtract',
        'max',
        'min',
        'number_of_unique_colors',
        'unique_colors',
        'intersection',
        'a_remove_b_colors',
        'b_remove_a_colors',
    ]
    transformation_weights = [20, 20, 20, 20, 20, 20, 20, 20, 20]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

    dataset_name = random.Random(seed + 1005).choice(DATASET_NAMES)

    instructions_add = [
        f'Add two {dataset_name} together',
        f'Add these {dataset_name}',
        f'Sum these {dataset_name}',
        f'{dataset_name}, perform addition',
        f'{dataset_name}, perform add',
        f'{dataset_name} add',
        f'{dataset_name} plus',
        f'{dataset_name} sum',
    ]

    instructions_subtract = [
        f'Subtract these {dataset_name}',
        f'{dataset_name}, perform subtraction',
        f'{dataset_name}, perform subtract',
        f'{dataset_name}, perform minus',
        f'{dataset_name} minus',
        f'{dataset_name} subtract',
    ]

    instructions_min = [
        f'{dataset_name}, perform min',
        f'{dataset_name}, perform minimum',
        f'{dataset_name} min',
        f'{dataset_name} minimum',
        f'Min of {dataset_name}',
        f'Minimum of {dataset_name}',
    ]

    instructions_max = [
        f'{dataset_name}, perform max',
        f'{dataset_name}, perform maximum',
        f'{dataset_name} max',
        f'{dataset_name} maximum',
        f'Max of {dataset_name}',
        f'Maximum of {dataset_name}',
    ]

    instructions_number_of_unique_colors = [
        f'{dataset_name}, number of unique colors',
        f'number of unique colors in {dataset_name}',
        f'how many unique colors are there in {dataset_name}',
        f'{dataset_name}, unique color count',
    ]

    instructions_unique_colors = [
        f'{dataset_name}, unique colors',
        f'unique colors in {dataset_name}',
        f'unique colors of {dataset_name}',
        f'what are the unique colors in {dataset_name}',
        f'{dataset_name}, unique color list',
        f'{dataset_name}, unique colors',
    ]

    instructions_intersection = [
        f'{dataset_name}, color intersection',
        f'intersection of colors in {dataset_name}',
        f'Intersection of colors in {dataset_name}',
        f'what are the shared colors between {dataset_name}',
        f'{dataset_name}, intersecting color list',
        f'{dataset_name}, intersecting colors',
        f'{dataset_name}, overlapping colors',
        f'{dataset_name}, color overlap',
    ]

    instructions_a_remove_b_colors = [
        f'{dataset_name}, Remove Histogram B colors from Histogram A',
        f'{dataset_name}, Remove Histogram-B colors from Histogram-A',
        f'{dataset_name}, remove histogram-b colors from histogram-a',
        f'{dataset_name}, remove histogram b colors from histogram a',
        f'{dataset_name}, Exclude Histogram B colors from Histogram A',
        f'{dataset_name}, Exclude Histogram-B colors from Histogram-A',
        f'{dataset_name}, exclude histogram-b colors from histogram-a',
        f'{dataset_name}, exclude histogram b colors from histogram a',
        f'{dataset_name}, Histogram A without colors of Histogram B',
        f'{dataset_name}, Histogram-A without colors of Histogram-B',
        f'{dataset_name}, histogram-a without colors of histogram-b',
        f'{dataset_name}, histogram a without colors of histogram b',
        f'{dataset_name}, Histogram A excluding Histogram B colors',
        f'{dataset_name}, Histogram-A excluding Histogram-B colors',
        f'{dataset_name}, histogram-a excluding histogram-b colors',
        f'{dataset_name}, histogram a excluding histogram b colors',
    ]

    instructions_b_remove_a_colors = [
        f'{dataset_name}, Remove Histogram A colors from Histogram B',
        f'{dataset_name}, Remove Histogram-A colors from Histogram-B',
        f'{dataset_name}, remove histogram-a colors from histogram-b',
        f'{dataset_name}, remove histogram a colors from histogram b',
        f'{dataset_name}, Exclude Histogram A colors from Histogram B',
        f'{dataset_name}, Exclude Histogram-A colors from Histogram-B',
        f'{dataset_name}, exclude histogram-a colors from histogram-b',
        f'{dataset_name}, exclude histogram a colors from histogram b',
        f'{dataset_name}, Histogram B without colors of Histogram A',
        f'{dataset_name}, Histogram-B without colors of Histogram-a',
        f'{dataset_name}, histogram-b without colors of histogram-a',
        f'{dataset_name}, histogram b without colors of histogram a',
        f'{dataset_name}, Histogram B excluding Histogram A colors',
        f'{dataset_name}, Histogram-B excluding Histogram-a colors',
        f'{dataset_name}, histogram-b excluding histogram-a colors',
        f'{dataset_name}, histogram b excluding histogram A colors',
    ]

    instructions = None
    if transformation_id == 'add':
        instructions = instructions_add
    elif transformation_id == 'subtract':
        instructions = instructions_subtract
    elif transformation_id == 'min':
        instructions = instructions_min
    elif transformation_id == 'max':
        instructions = instructions_max
    elif transformation_id == 'number_of_unique_colors':
        instructions = instructions_number_of_unique_colors
    elif transformation_id == 'unique_colors':
        instructions = instructions_unique_colors
    elif transformation_id == 'intersection':
        instructions = instructions_intersection
    elif transformation_id == 'a_remove_b_colors':
        instructions = instructions_a_remove_b_colors
    elif transformation_id == 'b_remove_a_colors':
        instructions = instructions_b_remove_a_colors
    else:
        raise Exception("Unreachable code reached")

    instruction = random.Random(seed + 1006).choice(instructions)

    max_color_value = 1600
    histogram0 = Histogram.create_random(seed + 1007, 1, 9, 1, max_color_value)
    histogram1 = Histogram.create_random(seed + 1020, 1, 9, 1, max_color_value)

    input = f'{histogram0.pretty()}\n{histogram1.pretty()}'

    output = None
    if transformation_id == 'add':
        output = histogram0.add(histogram1).pretty()
    elif transformation_id == 'subtract':
        output = histogram0.subtract_and_discard(histogram1).pretty()
    elif transformation_id == 'max':
        output = histogram0.max(histogram1).pretty()
    elif transformation_id == 'min':
        output = histogram0.min(histogram1).pretty()
    elif transformation_id == 'number_of_unique_colors':
        histogram = histogram0.add(histogram1)
        output = str(histogram.number_of_unique_colors())
    elif transformation_id == 'unique_colors':
        histogram = histogram0.add(histogram1)
        output = histogram.unique_colors_pretty()
    elif transformation_id == 'intersection':
        output = histogram0.color_intersection_pretty(histogram1)
    elif transformation_id == 'a_remove_b_colors':
        output = histogram0.remove_other_colors(histogram1).pretty()
    elif transformation_id == 'b_remove_a_colors':
        output = histogram1.remove_other_colors(histogram0).pretty()
    else:
        raise Exception("Unreachable code reached")

    sum_of_counters = histogram0.sum_of_counters() + histogram1.sum_of_counters()
    benchmark_histogram_size = histogram_total_to_string(sum_of_counters)
    benchmark_dataset_name = BENCHMARK_DATASET_NAME_TWO
    benchmark_id = f'dataset={benchmark_dataset_name} group={transformation_id} histogram_size={benchmark_histogram_size}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id,
    }
    return result_dict

def generate_dataset_item_list(seed: int) -> list[dict]:
    item = None
    if seed % 20 == 0:
        item = generate_one_histogram_dataset_item(seed)
    else:
        item = generate_two_histogram_dataset_item(seed)
    return [item]

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=3232003,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
