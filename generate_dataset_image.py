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
# IDEA: multiple size types. corpus: easy, medium, hard
# size10 images 1px to 10px
# size20 images 11px to 20px
# size30 images 21px to 30px
# size40 images 31px to 40px
# size50 images 41px to 50px
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
    min_image_size = 20
    max_image_size = 30

    input_formats = [
        'pixels', 
        'json'
    ]
    input_format_weights = [45, 45]
    input_format = random.Random(seed + 1001).choices(input_formats, weights=input_format_weights, k=1)[0]

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
    if input_format == 'pixels':
        name_input = random.Random(seed + 1002).choice(names_pixels)
    else:
        if input_format == 'json':
            name_input = random.Random(seed + 1003).choice(names_json)

    name_outputs = [
        'SIMONARCRLEIMAGE',
        'Simon-ARC-RLE-Image',
        'SimonsRLEImage',
    ]
    name_output = random.Random(seed + 1004).choice(name_outputs)

    instructions_input_output = [
        f'Serialize {name_input} to {name_output}',
        f'serialize {name_input} to {name_output}',
        f'convert {name_input} to {name_output}',
        f'Convert {name_input} to {name_output}',
        f'Transform {name_input} to {name_output}',
        f'transform {name_input} to {name_output}',
        f'Change {name_input} to {name_output}',
        f'change {name_input} to {name_output}',
        f'{name_input} to {name_output}',
        f'{name_output} from {name_input}',
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
    if input_format == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        input = ','.join(rows)
    else:
        if input_format == 'json':
            image_list = image.tolist()
            input = json.dumps(image_list, separators=(',', ':'))

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_deserialize_dataset_item(seed):
    """
    Convert from RLE representation to pixel representation.
    Transform the RLE representation to: histogram, flip, rotate, transpose.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    min_image_size = 1
    max_image_size = 10

    instruction_ids = [
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
    ]
    instruction_weights = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 200, 200, 200]
    instruction_id = random.Random(seed + 1001).choices(instruction_ids, weights=instruction_weights, k=1)[0]

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
    if instruction_id == 'pixels':
        name_output = random.Random(seed + 1002).choice(names_pixels)
    else:
        if instruction_id == 'json':
            name_output = random.Random(seed + 1003).choice(names_json)

    name_inputs = [
        'SIMONARCRLEIMAGE',
        'SIMONSARCRLEIMAGE',
        'SIMONSARCIMAGE',
        'SIMONARCIMAGE',
        'Simon-ARC-RLE-Image',
        'Simons-ARC-RLE-Image',
        'Simons-ARC-Image',
        'simons-arc-image',
        'SimonsArcRleImage',
        'SimonsRLEImage',
    ]
    name_input = random.Random(seed + 1004).choice(name_inputs)

    instructions_input_output = [
        f'Deserialize {name_input} to {name_output}',
        f'deserialize {name_input} to {name_output}',
        f'convert {name_input} to {name_output}',
        f'Convert {name_input} to {name_output}',
        f'Transform {name_input} to {name_output}',
        f'transform {name_input} to {name_output}',
        f'Change {name_input} to {name_output}',
        f'change {name_input} to {name_output}',
        f'{name_input} to {name_output}',
        f'{name_output} from {name_input}',
    ]

    instructions_histogram = [
        f'Histogram of deserialized {name_input}',
        f'histogram of deserialized {name_input}',
        f'Histogram after deserializing {name_input}',
        f'histogram after deserializing {name_input}',
        f'Histogram of {name_input}',
        f'histogram of {name_input}',
        f'Histogram of {name_input}',
        f'convert {name_input} and return the histogram',
        f'Convert {name_input} and return histogram',
        f'Process {name_input} and return the histogram',
        f'process {name_input} and return histogram',
    ]

    instructions_flipx = [
        f'FlipX {name_input}',
        f'Flip-X {name_input}',
        f'flipx {name_input}',
        f'convert {name_input} and return the flipx',
        f'process {name_input} and return flipx',
    ]

    instructions_flipy = [
        f'FlipY {name_input}',
        f'Flip-Y {name_input}',
        f'flipy {name_input}',
        f'convert {name_input} and return the flipy',
        f'process {name_input} and return flipy',
    ]

    instructions_transpose = [
        f'Transpose {name_input}',
        f'transpose {name_input}',
        f'{name_input} transposed',
        f'Process {name_input} and return the transposed',
        f'process {name_input} and return the transposed',
        f'Convert {name_input} and return the transposed',
        f'convert {name_input} and return the transposed',
    ]

    instructions_rotate_cw = [
        f'Rotate Clockwise {name_input}',
        f'Rotate clockwise {name_input}',
        f'Rotate clock-wise {name_input}',
        f'Rotate cw {name_input}',
        f'rotate CW {name_input}',
        f'CW rotate {name_input}',
        f'cw rotate {name_input}',
        f'Process {name_input} and return the clockwise rotated',
        f'process {name_input} and return the cw rotated',
    ]

    instructions_rotate_ccw = [
        f'Rotate CounterClockwise {name_input}',
        f'Rotate counterclockwise {name_input}',
        f'Rotate counter-clock-wise {name_input}',
        f'Rotate ccw {name_input}',
        f'rotate CCW {name_input}',
        f'CCW rotate {name_input}',
        f'ccw rotate {name_input}',
        f'Process {name_input} and return the counter clock wise rotated',
        f'process {name_input} and return the ccw rotated',
    ]

    instructions_rotate_180 = [
        f'Rotate 180 {name_input}',
        f'rotate 180 {name_input}',
        f'Half rotate {name_input}',
        f'Half a rotation of {name_input}',
        f'{name_input} rotated halfway',
        f'{name_input} rotated by 180 degrees',
    ]

    instructions_count_neighbors_with_same_color = [
        f'With {name_input}, 3x3 count neighbors with same color as center',
        f'With {name_input}, Number of neighbors with same color as center',
        f'{name_input}, 3x3 area, how many neighbors have the same color as center',
        f'{name_input}, 3x3 area, count neighbors with same color as center',
    ]

    instructions_all_neighbors_matching_center = [
        f'With {name_input}, all pixels inside 3x3 have same color as center',
        f'With {name_input}, 3x3 area, where all pixels have same color as center',
        f'{name_input}, 3x3 area, locations where all neighbors have the same color as center',
        f'{name_input}, 3x3 area, positions where all neighbors have the same color as center',
    ]

    pixels_with_k_matching_neighbors_k_parameter = random.Random(seed + 1005).randint(1, 8)
    instructions_pixels_with_k_matching_neighbors = [
        f'With {name_input}, where {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
        f'{name_input}, where {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
        f'{name_input}, where {pixels_with_k_matching_neighbors_k_parameter} of the 3x3 neighbors have the same color as the center pixel',
        f'{name_input}, identify pixels where exactly {pixels_with_k_matching_neighbors_k_parameter} neighbors have the same color as the center pixel',
    ]

    instructions_compress_x = [
        f'CompressX {name_input}',
        f'Compress X {name_input}',
        f'compress x {name_input}',
        f'Compress-X {name_input}',
        f'{name_input} Compress-X',
        f'{name_input} compress x',
        f'{name_input} remove duplicate adjacent columns',
        f'remove duplicate adjacent columns from {name_input}',
    ]

    instructions_compress_y = [
        f'CompressY {name_input}',
        f'Compress Y {name_input}',
        f'compress y {name_input}',
        f'Compress-Y {name_input}',
        f'{name_input} Compress-Y',
        f'{name_input} compress y',
        f'{name_input} remove duplicate adjacent rows',
        f'remove duplicate adjacent rows from {name_input}',
    ]

    instructions_compress_xy = [
        f'CompressXY {name_input}',
        f'compressxy {name_input}',
        f'Compress-XY {name_input}',
        f'Compress XY {name_input}',
        f'compress xy {name_input}',
        f'compress x and compress y {name_input}',
        f'compress x and y {name_input}',
        f'Compress X and Y {name_input}',
        f'{name_input} Compress-XY',
        f'{name_input} compress xy',
        f'{name_input} compressxy',
        f'{name_input} remove duplicate adjacent rows and columns',
        f'{name_input} remove duplicate adjacent columns and rows',
        f'remove duplicate adjacent rows and columns from {name_input}',
        f'remove duplicate adjacent columns and rows from {name_input}',
    ]

    instructions = instructions_input_output
    if instruction_id == 'histogram':
        instructions = instructions_histogram
    if instruction_id == 'flipx':
        instructions = instructions_flipx
    if instruction_id == 'flipy':
        instructions = instructions_flipy
    if instruction_id == 'transpose':
        instructions = instructions_transpose
    if instruction_id == 'rotate_cw':
        instructions = instructions_rotate_cw
    if instruction_id == 'rotate_ccw':
        instructions = instructions_rotate_ccw
    if instruction_id == 'rotate_180':
        instructions = instructions_rotate_180
    if instruction_id == 'count_neighbors_with_same_color':
        instructions = instructions_count_neighbors_with_same_color
    if instruction_id == 'all_neighbors_matching_center':
        instructions = instructions_all_neighbors_matching_center
    if instruction_id == 'pixels_with_k_matching_neighbors':
        instructions = instructions_pixels_with_k_matching_neighbors
    if instruction_id == 'compress_x':
        instructions = instructions_compress_x
    if instruction_id == 'compress_y':
        instructions = instructions_compress_y
    if instruction_id == 'compress_xy':
        instructions = instructions_compress_xy

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, image = generate_rle_string(
        seed=seed + 1006, 
        min_image_size=min_image_size, 
        max_image_size=max_image_size
    )

    output = None
    if instruction_id == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        output = ','.join(rows)
    elif instruction_id == 'json':
        image_list = image.tolist()
        output = json.dumps(image_list, separators=(',', ':'))
    elif instruction_id == 'histogram':
        histogram = Histogram.create_with_image(image)
        output = histogram.pretty()
    elif instruction_id == 'flipx':
        flipped_image = image[:, ::-1]
        output_rle_string = serialize(flipped_image)
        output = output_rle_string
    elif instruction_id == 'flipy':
        flipped_image = image[::-1, :]
        output_rle_string = serialize(flipped_image)
        output = output_rle_string
    elif instruction_id == 'transpose':
        transposed_image = image.transpose()
        output_rle_string = serialize(transposed_image)
        output = output_rle_string
    elif instruction_id == 'rotate_cw':
        new_image = image_rotate_cw(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'rotate_ccw':
        new_image = image_rotate_ccw(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'rotate_180':
        new_image = image_rotate_180(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'count_neighbors_with_same_color':
        new_image = count_neighbors_with_same_color_nowrap(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'all_neighbors_matching_center':
        new_image = all_neighbors_matching_center_nowrap(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'pixels_with_k_matching_neighbors':
        new_image = pixels_with_k_matching_neighbors_nowrap(image, pixels_with_k_matching_neighbors_k_parameter)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'compress_x':
        new_image = compress_x(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'compress_y':
        new_image = compress_y(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    elif instruction_id == 'compress_xy':
        new_image = compress_xy(image)
        output_rle_string = serialize(new_image)
        output = output_rle_string
    else:
        raise Exception("Unreachable code reached")

    dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=400501):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        if i % 40 == 0:
            item = generate_serialize_dataset_item(seed_start + i)
        else:
            item = generate_deserialize_dataset_item(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
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

