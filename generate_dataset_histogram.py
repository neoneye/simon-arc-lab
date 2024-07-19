# IDEA: Exercise much bigger values up to 900. Currently have been exercising the range 1-800.
#
# IDEA: What are the most popular colors in the histogram?
# IDEA: What are the least popular colors in the histogram?
import json
import os
import random
from simon_arc_lab.histogram import *

name_formats = [
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

    name_format = random.Random(seed + 1005).choice(name_formats)

    instructions_number_of_unique_colors = [
        f'{name_format}, number of unique colors',
        f'number of unique colors in {name_format}',
        f'how many unique colors are there in {name_format}',
        f'{name_format}, unique color count',
    ]

    instructions_unique_colors = [
        f'{name_format}, unique colors',
        f'unique colors in {name_format}',
        f'unique colors of {name_format}',
        f'what are the unique colors in {name_format}',
        f'{name_format}, unique color list',
        f'{name_format}, unique colors',
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

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

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

    name_format = random.Random(seed + 1005).choice(name_formats)

    instructions_add = [
        f'Add two {name_format} together',
        f'Add these {name_format}',
        f'Sum these {name_format}',
        f'{name_format}, perform addition',
        f'{name_format}, perform add',
        f'{name_format} add',
        f'{name_format} plus',
        f'{name_format} sum',
    ]

    instructions_subtract = [
        f'Subtract these {name_format}',
        f'{name_format}, perform subtraction',
        f'{name_format}, perform subtract',
        f'{name_format}, perform minus',
        f'{name_format} minus',
        f'{name_format} subtract',
    ]

    instructions_min = [
        f'{name_format}, perform min',
        f'{name_format}, perform minimum',
        f'{name_format} min',
        f'{name_format} minimum',
        f'Min of {name_format}',
        f'Minimum of {name_format}',
    ]

    instructions_max = [
        f'{name_format}, perform max',
        f'{name_format}, perform maximum',
        f'{name_format} max',
        f'{name_format} maximum',
        f'Max of {name_format}',
        f'Maximum of {name_format}',
    ]

    instructions_number_of_unique_colors = [
        f'{name_format}, number of unique colors',
        f'number of unique colors in {name_format}',
        f'how many unique colors are there in {name_format}',
        f'{name_format}, unique color count',
    ]

    instructions_unique_colors = [
        f'{name_format}, unique colors',
        f'unique colors in {name_format}',
        f'unique colors of {name_format}',
        f'what are the unique colors in {name_format}',
        f'{name_format}, unique color list',
        f'{name_format}, unique colors',
    ]

    instructions_intersection = [
        f'{name_format}, color intersection',
        f'intersection of colors in {name_format}',
        f'Intersection of colors in {name_format}',
        f'what are the shared colors between {name_format}',
        f'{name_format}, intersecting color list',
        f'{name_format}, intersecting colors',
        f'{name_format}, overlapping colors',
        f'{name_format}, color overlap',
    ]

    instructions_a_remove_b_colors = [
        f'{name_format}, Remove Histogram B colors from Histogram A',
        f'{name_format}, Remove Histogram-B colors from Histogram-A',
        f'{name_format}, remove histogram-b colors from histogram-a',
        f'{name_format}, remove histogram b colors from histogram a',
        f'{name_format}, Exclude Histogram B colors from Histogram A',
        f'{name_format}, Exclude Histogram-B colors from Histogram-A',
        f'{name_format}, exclude histogram-b colors from histogram-a',
        f'{name_format}, exclude histogram b colors from histogram a',
        f'{name_format}, Histogram A without colors of Histogram B',
        f'{name_format}, Histogram-A without colors of Histogram-B',
        f'{name_format}, histogram-a without colors of histogram-b',
        f'{name_format}, histogram a without colors of histogram b',
        f'{name_format}, Histogram A excluding Histogram B colors',
        f'{name_format}, Histogram-A excluding Histogram-B colors',
        f'{name_format}, histogram-a excluding histogram-b colors',
        f'{name_format}, histogram a excluding histogram b colors',
    ]

    instructions_b_remove_a_colors = [
        f'{name_format}, Remove Histogram A colors from Histogram B',
        f'{name_format}, Remove Histogram-A colors from Histogram-B',
        f'{name_format}, remove histogram-a colors from histogram-b',
        f'{name_format}, remove histogram a colors from histogram b',
        f'{name_format}, Exclude Histogram A colors from Histogram B',
        f'{name_format}, Exclude Histogram-A colors from Histogram-B',
        f'{name_format}, exclude histogram-a colors from histogram-b',
        f'{name_format}, exclude histogram a colors from histogram b',
        f'{name_format}, Histogram B without colors of Histogram A',
        f'{name_format}, Histogram-B without colors of Histogram-a',
        f'{name_format}, histogram-b without colors of histogram-a',
        f'{name_format}, histogram b without colors of histogram a',
        f'{name_format}, Histogram B excluding Histogram A colors',
        f'{name_format}, Histogram-B excluding Histogram-a colors',
        f'{name_format}, histogram-b excluding histogram-a colors',
        f'{name_format}, histogram b excluding histogram A colors',
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

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=900000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        if i % 20 == 0:
            item = generate_one_histogram_dataset_item(seed_start + i)
        else:
            item = generate_two_histogram_dataset_item(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    return dataset

dataset = generate_dataset(
    max_num_samples=100000,
    max_byte_size=1024*1024*20,
)

# Save dataset to file
filename = 'dataset_histogram.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

