# IDEA: Exercise with bigger image. Currently have been exercising the range 1-10. Somewhat in the range 1-20.
#
# IDEA: Exercise rotate with bigger image. Currently have been exercising the range 1-20. And needs training in the range 21-30.
#
# IDEA: transformation "replace colors" of the image
#
# IDEA: transformation "transpose" the image
#
# IDEA: with "rot" prefix, then the image is to be rotated 90 degrees clockwise
#
# IDEA: is there a pixel above, below, left, right, that is the same as the center pixel. All the pixels in the 3x3 area.
# wraparound, wrapx, wrapy, nowrap
#
# IDEA: number of identical neighboring pixels in the 3x3 area in diagonal corners. Max 4 pixels can be the same as the center.
# IDEA: number of identical neighboring pixels in the 3x3 area in adjacent to center. Max 4 pixels can be the same as the center.
# wraparound, wrapx, wrapy, nowrap
#
# IDEA: transformation "rotate" the image
#
# IDEA: transformation "flip" the image
#
# IDEA: auto detect what image format it is, and convert it to RLE format.
#
# IDEA: deserialize images with "rot" prefix, then the image is to be rotated 90 degrees clockwise
#
# IDEA: scale the image by 2, 3, 4, 5, 6
# IDEA: take N top/bottom rows, N left/right columns
# IDEA: remove N top/bottom rows, N left/right columns
import json
import os
import random
import numpy as np
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_util import *
from simon_arc_lab.histogram import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.benchmark import *

DATASET_NAMES = [
    'SIMONARCRLEIMAGE',
    'SIMONSARCRLEIMAGE',
    'SIMONSARCIMAGE',
    'SIMONARCIMAGE',
    'Simon-ARC-RLE-Image',
    'Simons-ARC-RLE-Image',
    'Simons-ARC-Image',
    'simons-arc-image',
    'SimonsArcRleImage',
    'SimonsRLEImage'
]

def generate_rle_string(seed, min_image_size=1, max_image_size=100):
    """
    Generate a RLE string of a random image.

    :param seed: The seed for the random number generator
    :param min_image_size: The minimum size of the image
    :param max_image_size: The maximum size of the image
    :return: A tuple of a randomly generated RLE string and the corresponding image
    """

    image = image_create_random_advanced(seed, min_image_size, max_image_size, min_image_size, max_image_size)

    rle_string = serialize(image)
    
    return (rle_string, image)

def generate_serialize_dataset_item(seed):
    """
    Convert from pixel representation to RLE representation.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 30

    transformation_ids = [
        'pixels', 
        'json'
    ]
    transformation_weights = [45, 45]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]

    names_pixels = [
        'Pixels',
        'pixels',
        'Digits',
        'digits',
        'Symbols',
        'symbols',
        'String',
        'string',
    ]
    names_json = [
        'Json',
        'json',
        'JSON',
    ]

    name_input = None
    if transformation_id == 'pixels':
        name_input = random.Random(seed + 1002).choice(names_pixels)
    elif transformation_id == 'json':
        name_input = random.Random(seed + 1003).choice(names_json)
    else:
        raise Exception("Unreachable code reached")

    dataset_name = random.Random(seed + 1004).choice(DATASET_NAMES)

    instructions_input_output = [
        f'Serialize {name_input} to {dataset_name}',
        f'serialize {name_input} to {dataset_name}',
        f'convert {name_input} to {dataset_name}',
        f'Convert {name_input} to {dataset_name}',
        f'Transform {name_input} to {dataset_name}',
        f'transform {name_input} to {dataset_name}',
        f'Change {name_input} to {dataset_name}',
        f'change {name_input} to {dataset_name}',
        f'{name_input} to {dataset_name}',
        f'{dataset_name} from {name_input}',
    ]

    instructions = instructions_input_output

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, image = generate_rle_string(
        seed=seed + 1006, 
        min_image_size=min_image_size,
        max_image_size=max_image_size
    )

    output = rle_string

    input = None
    if transformation_id == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        input = ','.join(rows)
    elif transformation_id == 'json':
        image_list = image.tolist()
        input = json.dumps(image_list, separators=(',', ':'))
    else:
        raise Exception("Unreachable code reached")

    width, height = image.shape[1], image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_id = f'dataset=image_serialize group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_deserialize_dataset_item(seed_start, item_index):
    """
    Convert from RLE representation to pixel representation.
    Transform the RLE representation to: histogram, flip, rotate, transpose.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 21

    seed = seed_start + item_index

    # image_seed = seed + 1006 
    image_seed = seed_start + (item_index // 2)  # Use the same image for rotate_cw and rotate_ccw
    rle_string, image = generate_rle_string(
        seed=image_seed, 
        min_image_size=min_image_size, 
        max_image_size=max_image_size
    )
    image_height = image.shape[0]
    image_width = image.shape[1]

    transformation_ids = [
        'pixels', 
        'json',
        'histogram',
        'flipx',
        'flipy',
        'transpose',
        'rotate_cw',
        'rotate_ccw',
        'rotate_180',
        'count_neighbors_with_same_color',
        'all_neighbors_matching_center',
        'pixels_with_k_matching_neighbors',
        'compress_x',
        'compress_y',
        'compress_xy',
        'translate_x_minus1',
        'translate_x_plus1',
        'translate_y_minus1',
        'translate_y_plus1',
        'get_row_as_list',
        'get_column_as_list',
    ]
    # transformation_weights = [0, 0, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # transformation_weights = [0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    transformation_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10]
    # transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]
    transformation_id = 'rotate_cw'
    if item_index % 2 == 0:
        transformation_id = 'rotate_ccw'

    names_pixels = [
        'Pixels',
        'pixels',
        'Digits',
        'digits',
        'Symbols',
        'symbols',
        'String',
        'string',
    ]
    names_json = [
        'Json',
        'json',
        'JSON',
    ]

    name_output = None
    if transformation_id == 'pixels':
        name_output = random.Random(seed + 1002).choice(names_pixels)
    else:
        if transformation_id == 'json':
            name_output = random.Random(seed + 1003).choice(names_json)

    dataset_name = random.Random(seed + 1004).choice(DATASET_NAMES)

    instructions_input_output = [
        f'Deserialize {dataset_name} to {name_output}',
        f'deserialize {dataset_name} to {name_output}',
        f'convert {dataset_name} to {name_output}',
        f'Convert {dataset_name} to {name_output}',
        f'Transform {dataset_name} to {name_output}',
        f'transform {dataset_name} to {name_output}',
        f'Change {dataset_name} to {name_output}',
        f'change {dataset_name} to {name_output}',
        f'{dataset_name} to {name_output}',
        f'{name_output} from {dataset_name}',
    ]

    instructions_histogram = [
        f'Histogram of deserialized {dataset_name}',
        f'histogram of deserialized {dataset_name}',
        f'Histogram after deserializing {dataset_name}',
        f'histogram after deserializing {dataset_name}',
        f'Histogram of {dataset_name}',
        f'histogram of {dataset_name}',
        f'Histogram of {dataset_name}',
        f'convert {dataset_name} and return the histogram',
        f'Convert {dataset_name} and return histogram',
        f'Process {dataset_name} and return the histogram',
        f'process {dataset_name} and return histogram',
    ]

    instructions_flipx = [
        f'FlipX {dataset_name}',
        f'Flip-X {dataset_name}',
        f'flipx {dataset_name}',
        f'convert {dataset_name} and return the flipx',
        f'process {dataset_name} and return flipx',
    ]

    instructions_flipy = [
        f'FlipY {dataset_name}',
        f'Flip-Y {dataset_name}',
        f'flipy {dataset_name}',
        f'convert {dataset_name} and return the flipy',
        f'process {dataset_name} and return flipy',
    ]

    instructions_transpose = [
        f'Transpose {dataset_name}',
        f'transpose {dataset_name}',
        f'{dataset_name} transposed',
        f'Process {dataset_name} and return the transposed',
        f'process {dataset_name} and return the transposed',
        f'Convert {dataset_name} and return the transposed',
        f'convert {dataset_name} and return the transposed',
    ]

    instructions_rotate_cw = [
        f'Rotate Clockwise {dataset_name}',
        f'Rotate clockwise {dataset_name}',
        f'Rotate clock-wise {dataset_name}',
        f'Rotate cw {dataset_name}',
        f'rotate CW {dataset_name}',
        f'CW rotate {dataset_name}',
        f'cw rotate {dataset_name}',
        f'Process {dataset_name} and return the clockwise rotated',
        f'process {dataset_name} and return the cw rotated',
    ]

    instructions_rotate_ccw = [
        f'Rotate CounterClockwise {dataset_name}',
        f'Rotate counterclockwise {dataset_name}',
        f'Rotate counter-clock-wise {dataset_name}',
        f'Rotate ccw {dataset_name}',
        f'rotate CCW {dataset_name}',
        f'CCW rotate {dataset_name}',
        f'ccw rotate {dataset_name}',
        f'Process {dataset_name} and return the counter clock wise rotated',
        f'process {dataset_name} and return the ccw rotated',
    ]

    instructions_rotate_180 = [
        f'Rotate 180 {dataset_name}',
        f'rotate 180 {dataset_name}',
        f'Half rotate {dataset_name}',
        f'Half a rotation of {dataset_name}',
        f'{dataset_name} rotated halfway',
        f'{dataset_name} rotated by 180 degrees',
    ]

    instructions_count_neighbors_with_same_color = [
        f'With {dataset_name}, 3x3 count neighbors with same color as center',
        f'With {dataset_name}, Number of neighbors with same color as center',
        f'{dataset_name}, 3x3 area, how many neighbors have the same color as center',
        f'{dataset_name}, 3x3 area, count neighbors with same color as center',
    ]

    instructions_all_neighbors_matching_center = [
        f'With {dataset_name}, all pixels inside 3x3 have same color as center',
        f'With {dataset_name}, 3x3 area, where all pixels have same color as center',
        f'{dataset_name}, 3x3 area, locations where all neighbors have the same color as center',
        f'{dataset_name}, 3x3 area, positions where all neighbors have the same color as center',
    ]

    pixels_with_k_matching_neighbors_k_parameter = random.Random(seed + 1005).randint(1, 8)
    instructions_pixels_with_k_matching_neighbors = [
        f'With {dataset_name}, where {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
        f'{dataset_name}, where {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
        f'{dataset_name}, where {pixels_with_k_matching_neighbors_k_parameter} of the 3x3 neighbors have the same color as the center pixel',
        f'{dataset_name}, identify pixels where exactly {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
    ]

    instructions_compress_x = [
        f'CompressX {dataset_name}',
        f'Compress X {dataset_name}',
        f'compress x {dataset_name}',
        f'Compress-X {dataset_name}',
        f'{dataset_name} Compress-X',
        f'{dataset_name} compress x',
        f'{dataset_name} remove duplicate adjacent columns',
        f'remove duplicate adjacent columns from {dataset_name}',
    ]

    instructions_compress_y = [
        f'CompressY {dataset_name}',
        f'Compress Y {dataset_name}',
        f'compress y {dataset_name}',
        f'Compress-Y {dataset_name}',
        f'{dataset_name} Compress-Y',
        f'{dataset_name} compress y',
        f'{dataset_name} remove duplicate adjacent rows',
        f'remove duplicate adjacent rows from {dataset_name}',
    ]

    instructions_compress_xy = [
        f'CompressXY {dataset_name}',
        f'compressxy {dataset_name}',
        f'Compress-XY {dataset_name}',
        f'Compress XY {dataset_name}',
        f'compress xy {dataset_name}',
        f'compress x and compress y {dataset_name}',
        f'compress x and y {dataset_name}',
        f'Compress X and Y {dataset_name}',
        f'{dataset_name} Compress-XY',
        f'{dataset_name} compress xy',
        f'{dataset_name} compressxy',
        f'{dataset_name} remove duplicate adjacent rows and columns',
        f'{dataset_name} remove duplicate adjacent columns and rows',
        f'remove duplicate adjacent rows and columns from {dataset_name}',
        f'remove duplicate adjacent columns and rows from {dataset_name}',
    ]

    instructions_translate_x_minus1 = [
        f'Translate x minus 1, {dataset_name}',
        f'Translate x-1 {dataset_name}',
        f'move left by 1 pixel {dataset_name}',
        f'{dataset_name}, translate x by -1',
        f'{dataset_name}, return translated x-1',
        f'{dataset_name}, move left by 1 pixel',
    ]
    instructions_translate_x_plus1 = [
        f'Translate x plus 1, {dataset_name}',
        f'Translate x+1 {dataset_name}',
        f'move right by 1 pixel {dataset_name}',
        f'{dataset_name}, translate x by +1',
        f'{dataset_name}, return translated x+1',
        f'{dataset_name}, move right by 1 pixel',
    ]
    instructions_translate_y_minus1 = [
        f'Translate y minus 1, {dataset_name}',
        f'Translate y-1 {dataset_name}',
        f'move up by 1 pixel {dataset_name}',
        f'{dataset_name}, translate y by -1',
        f'{dataset_name}, return translated y-1',
        f'{dataset_name}, move up by 1 pixel',
    ]
    instructions_translate_y_plus1 = [
        f'Translate y plus 1, {dataset_name}',
        f'Translate y+1 {dataset_name}',
        f'move down by 1 pixel {dataset_name}',
        f'{dataset_name}, translate y by +1',
        f'{dataset_name}, return translated y+1',
        f'{dataset_name}, move down by 1 pixel',
    ]

    get_row_y_as_list_y_parameter = random.Random(seed + 1006).randint(0, image_height - 1)
    instructions_get_row_as_list = [
        f'Get pixels from row {get_row_y_as_list_y_parameter}, {dataset_name}',
        f'Get digits from row {get_row_y_as_list_y_parameter}, {dataset_name}',
        f'Get symbols from row {get_row_y_as_list_y_parameter}, {dataset_name}',
        f'Get row {get_row_y_as_list_y_parameter} as pixels, {dataset_name}',
        f'Get row {get_row_y_as_list_y_parameter} as digits, {dataset_name}',
        f'Get row {get_row_y_as_list_y_parameter} as symbols, {dataset_name}',
        f'{dataset_name}, Get pixels from row {get_row_y_as_list_y_parameter}',
        f'{dataset_name}, Get digits from row {get_row_y_as_list_y_parameter}',
        f'{dataset_name}, Get symbols from row {get_row_y_as_list_y_parameter}',
        f'{dataset_name}, Get row {get_row_y_as_list_y_parameter} as pixels',
        f'{dataset_name}, Get row {get_row_y_as_list_y_parameter} as digits',
        f'{dataset_name}, Get row {get_row_y_as_list_y_parameter} as symbols',
    ]

    get_column_x_as_list_x_parameter = random.Random(seed + 1007).randint(0, image_width - 1)
    instructions_get_column_as_list = [
        f'Get pixels from column {get_column_x_as_list_x_parameter}, {dataset_name}',
        f'Get digits from column {get_column_x_as_list_x_parameter}, {dataset_name}',
        f'Get symbols from column {get_column_x_as_list_x_parameter}, {dataset_name}',
        f'Get column {get_column_x_as_list_x_parameter} as pixels, {dataset_name}',
        f'Get column {get_column_x_as_list_x_parameter} as digits, {dataset_name}',
        f'Get column {get_column_x_as_list_x_parameter} as symbols, {dataset_name}',
        f'{dataset_name}, Get pixels from column {get_column_x_as_list_x_parameter}',
        f'{dataset_name}, Get digits from column {get_column_x_as_list_x_parameter}',
        f'{dataset_name}, Get symbols from column {get_column_x_as_list_x_parameter}',
        f'{dataset_name}, Get column {get_column_x_as_list_x_parameter} as pixels',
        f'{dataset_name}, Get column {get_column_x_as_list_x_parameter} as digits',
        f'{dataset_name}, Get column {get_column_x_as_list_x_parameter} as symbols',
    ]

    instructions = None
    if transformation_id == 'pixels':
        instructions = instructions_input_output
    elif transformation_id == 'json':
        instructions = instructions_input_output
    elif transformation_id == 'histogram':
        instructions = instructions_histogram
    elif transformation_id == 'flipx':
        instructions = instructions_flipx
    elif transformation_id == 'flipy':
        instructions = instructions_flipy
    elif transformation_id == 'transpose':
        instructions = instructions_transpose
    elif transformation_id == 'rotate_cw':
        instructions = instructions_rotate_cw
    elif transformation_id == 'rotate_ccw':
        instructions = instructions_rotate_ccw
    elif transformation_id == 'rotate_180':
        instructions = instructions_rotate_180
    elif transformation_id == 'count_neighbors_with_same_color':
        instructions = instructions_count_neighbors_with_same_color
    elif transformation_id == 'all_neighbors_matching_center':
        instructions = instructions_all_neighbors_matching_center
    elif transformation_id == 'pixels_with_k_matching_neighbors':
        instructions = instructions_pixels_with_k_matching_neighbors
    elif transformation_id == 'compress_x':
        instructions = instructions_compress_x
    elif transformation_id == 'compress_y':
        instructions = instructions_compress_y
    elif transformation_id == 'compress_xy':
        instructions = instructions_compress_xy
    elif transformation_id == 'translate_x_minus1':
        instructions = instructions_translate_x_minus1
    elif transformation_id == 'translate_x_plus1':
        instructions = instructions_translate_x_plus1
    elif transformation_id == 'translate_y_minus1':
        instructions = instructions_translate_y_minus1
    elif transformation_id == 'translate_y_plus1':
        instructions = instructions_translate_y_plus1
    elif transformation_id == 'get_row_as_list':
        instructions = instructions_get_row_as_list
    elif transformation_id == 'get_column_as_list':
        instructions = instructions_get_column_as_list
    else:
        raise Exception("Unreachable code reached")


    instruction = random.Random(seed + 1005).choice(instructions)

    output = None
    if transformation_id == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        output = ','.join(rows)
    elif transformation_id == 'json':
        image_list = image.tolist()
        output = json.dumps(image_list, separators=(',', ':'))
    elif transformation_id == 'histogram':
        histogram = Histogram.create_with_image(image)
        output = histogram.pretty()
    elif transformation_id == 'flipx':
        flipped_image = image[:, ::-1]
        output_rle_string = serialize(flipped_image)
        output = output_rle_string
    elif transformation_id == 'flipy':
        flipped_image = image[::-1, :]
        output_rle_string = serialize(flipped_image)
        output = output_rle_string
    elif transformation_id == 'transpose':
        transposed_image = image.transpose()
        output_rle_string = serialize(transposed_image)
        output = output_rle_string
    elif transformation_id == 'rotate_cw':
        new_image = image_rotate_cw(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'rotate_ccw':
        new_image = image_rotate_ccw(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'rotate_180':
        new_image = image_rotate_180(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'count_neighbors_with_same_color':
        new_image = count_neighbors_with_same_color_nowrap(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'all_neighbors_matching_center':
        new_image = all_neighbors_matching_center_nowrap(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'pixels_with_k_matching_neighbors':
        new_image = pixels_with_k_matching_neighbors_nowrap(image, pixels_with_k_matching_neighbors_k_parameter)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'compress_x':
        new_image = compress_x(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'compress_y':
        new_image = compress_y(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'compress_xy':
        new_image = compress_xy(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'translate_x_minus1':
        new_image = image_translate_wrap(image, -1, 0)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'translate_x_plus1':
        new_image = image_translate_wrap(image, 1, 0)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'translate_y_minus1':
        new_image = image_translate_wrap(image, 0, -1)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'translate_y_plus1':
        new_image = image_translate_wrap(image, 0, 1)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif transformation_id == 'get_row_as_list':
        pixel_list = image_get_row_as_list(image, get_row_y_as_list_y_parameter)
        output = ''.join(map(str, pixel_list))
    elif transformation_id == 'get_column_as_list':
        pixel_list = image_get_column_as_list(image, get_column_x_as_list_x_parameter)
        output = ''.join(map(str, pixel_list))
    else:
        raise Exception("Unreachable code reached")

    width, height = image.shape[1], image.shape[0]
    benchmark_width = image_size1d_to_string(width)
    benchmark_height = image_size1d_to_string(height)
    benchmark_id = f'dataset=image_deserialize group={transformation_id} image_width={benchmark_width} image_height={benchmark_height}'

    result_dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=3900000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        # if i % 100 == 0:
        #     item = generate_serialize_dataset_item(seed_start + i)
        # else:
        #     item = generate_deserialize_dataset_item(seed_start, i)
        item = generate_deserialize_dataset_item(seed_start, i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    random.Random(seed_start).shuffle(dataset)
    return dataset

dataset = generate_dataset(
    max_num_samples=100000,
    max_byte_size=1024*1024*60,
)

# Save dataset to file
filename = 'dataset_image.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

