# IDEA: can it be scaled down by factor 2, 3, 4, 5 without loss
# IDEA: Merge two encoded rows into one: "a3b2 + c4d1" = "a3b2c4d1"
# IDEA: crop
# IDEA: find pattern
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import os
import random
from simon_arc_lab.image_util import image_create
from simon_arc_lab.histogram import *
from simon_arc_lab.rle.deserialize import decode_rle_row_inner
from simon_arc_lab.rle.serialize import rle_serialize_line_inner
from simon_arc_lab.list_util import list_compress, list_scaleup
from simon_arc_lab.benchmark import *
from simon_arc_dataset.dataset_generator import *

SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_rle.jsonl')

def generate_rle_string_compacted(string_length=10, pixel_length=50, seed=None):
    """
    Generate a random RLE string of the specified length.
    The same color may appear multiple times, however it's compacted to its shortest form.

    :param string_length: The desired length of the RLE string
    :param pixel_length: The desired length of the pixel array
    :param seed: The seed for the random number generator
    :return: A tuple of a randomly generated RLE string and the corresponding pixel array
    """
    if seed is not None:
        random.seed(seed)

    rle_string = ''
    pixels = []
    while len(rle_string) < string_length and len(pixels) < pixel_length:
        digit = str(random.randint(0, 9))
        run_length = random.randint(1, 27)

        if run_length > 1:
            alpha_char = chr(ord('a') + (run_length - 2))
            rle_string += alpha_char + digit
        else:
            rle_string += digit

        pixels = decode_rle_row_inner(rle_string)

    compact_rle_string = rle_serialize_line_inner(pixels)

    return (compact_rle_string, pixels)

def generate_rle_string_noncompacted(string_length=10, pixel_length=50, seed=None):
    """
    Generate a random RLE string of the specified length. 
    The same color may appear multiple times.

    :param string_length: The desired length of the RLE string
    :param pixel_length: The desired length of the pixel array
    :param seed: The seed for the random number generator
    :return: A tuple of a randomly generated RLE string and the corresponding pixel array
    """
    if seed is not None:
        random.seed(seed)

    rle_string = ''
    pixels = []
    while len(rle_string) < string_length and len(pixels) < pixel_length:
        digit = str(random.randint(0, 9))
        run_length = random.randint(1, 27)

        if run_length > 1:
            alpha_char = chr(ord('a') + (run_length - 2))
            rle_string += alpha_char + digit
        else:
            rle_string += digit

        pixels = decode_rle_row_inner(rle_string)

    return (rle_string, pixels)


def generate_serialize_dataset_item(seed):
    string_length = 50 
    max_pixel_length = 100 
    pixel_length = random.Random(seed + 1000).randint(1, max_pixel_length)

    transformation_ids = [
        'pixels', 
        'json'
    ]
    transformation_weights = [0.5, 0.5]
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

    name_outputs = [
        'SIMONARCRLEROW',
        'Simon-ARC-RLE-Row',
        'SimonsRLERow',
    ]
    name_output = random.Random(seed + 1004).choice(name_outputs)

    instructions = [
        f'Serialize {name_input} to {name_output}',
        f'Serialize {name_input} to {name_output}',
        f'convert {name_input} to {name_output}',
        f'Convert {name_input} to {name_output}',
        f'Transform {name_input} to {name_output}',
        f'transform {name_input} to {name_output}',
        f'Change {name_input} to {name_output}',
        f'change {name_input} to {name_output}',
        f'{name_input} to {name_output}',
        f'{name_output} from {name_input}',
    ]

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, pixels = generate_rle_string_compacted(string_length=string_length, seed=seed + 1006, pixel_length=pixel_length)

    input = None
    if transformation_id == 'pixels':
        input = ''.join(map(str, pixels))
    elif transformation_id == 'json':
        input = json.dumps(list(pixels), separators=(',', ':'))
    else:
        raise Exception("Unreachable code reached")

    benchmark_length = image_size1d_to_string(pixel_length)
    benchmark_id = f'dataset=rle_serialize group={transformation_id} pixel_length={benchmark_length}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': rle_string,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_deserialize_dataset_item(seed):
    string_length = 50 
    max_pixel_length = 100 
    pixel_length = random.Random(seed + 1000).randint(1, max_pixel_length)

    transformation_ids = [
        'pixels', 
        'json',
        'length',
        'histogram',
        'reverse',
        'compress',
        'scaleup',
    ]
    # transformation_weights = [45, 45, 10, 30, 20, 20, 20]
    transformation_weights = [4, 5, 4, 100, 20, 20, 20]
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

    name_output = None
    if transformation_id == 'pixels':
        name_output = random.Random(seed + 1002).choice(names_pixels)
    else:
        if transformation_id == 'json':
            name_output = random.Random(seed + 1003).choice(names_json)

    scaleup_factor = random.Random(seed + 1004).randint(2, 5)

    name_inputs = [
        'SIMONARCRLEROW',
        'Simon-ARC-RLE-Row',
        'SimonsRLERow',
        'Simons-RLE-Row',
        'simons-Arc-Rle-row',
        'simon-Arc-Rle-row',
    ]
    name_input = random.Random(seed + 1005).choice(name_inputs)

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

    instructions_length = [
        f'Length of deserialized {name_input}',
        f'length of deserialized {name_input}',
        f'Length after deserializing {name_input}',
        f'length after deserializing {name_input}',
        f'Pixel count of {name_input}',
        f'pixel count of {name_input}',
        f'Number of pixels of {name_input}',
        f'convert {name_input} and return number of pixels',
        f'Convert {name_input} and return number of pixels',
        f'Process {name_input} and return number of pixels',
        f'process {name_input} and return number of pixels',
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
        f'{name_input} histogram',
        f'{name_input} Histogram',
        f'{name_input} compute histogram',
        f'{name_input} determine histogram',
        f'With {name_input} find the histogram',
        f'With {name_input} populate a histogram',
    ]

    instructions_reverse = [
        f'Reverse the {name_input}',
        f'reverse the {name_input}',
        f'Flipx {name_input}',
        f'flipx {name_input}',
        f'Flip-x {name_input}',
        f'flip-x {name_input}',
    ]

    instructions_compress = [
        f'Compress the {name_input}',
        f'compress the {name_input}',
        f'Compress {name_input}',
        f'compress {name_input}',
        f'Remove repeated pixels from {name_input}',
        f'remove repeated pixels from {name_input}',
    ]

    instructions_scaleup = [
        f'Scale up by {scaleup_factor} with the {name_input} data',
        f'scaleup by factor {scaleup_factor} with the {name_input} string',
        f'Scale up the {name_input} by {scaleup_factor}',
        f'scale up the {name_input} by {scaleup_factor}',
        f'ScaleUp {name_input} by factor {scaleup_factor}',
        f'scale-up {name_input} by factor {scaleup_factor}',
        f'Process {name_input} and apply scaleup by factor {scaleup_factor}',
        f'Process {name_input} and apply scale-up by factor {scaleup_factor}',
    ]

    instructions = instructions_input_output
    if transformation_id == 'length':
        instructions = instructions_length
    if transformation_id == 'histogram':
        instructions = instructions_histogram
    if transformation_id == 'reverse':
        instructions = instructions_reverse
    if transformation_id == 'compress':
        instructions = instructions_compress
    if transformation_id == 'scaleup':
        instructions = instructions_scaleup

    instruction = random.Random(seed + 1006).choice(instructions)

    rle_string, pixels = generate_rle_string_noncompacted(string_length=string_length, seed=seed + 1007, pixel_length=pixel_length)

    output = None
    if transformation_id == 'pixels':
        output = ''.join(map(str, pixels))
    elif transformation_id == 'json':
        output = json.dumps(list(pixels), separators=(',', ':'))
    elif transformation_id == 'length':
        output = str(len(pixels))
    elif transformation_id == 'histogram':
        image = image_create(1, len(pixels), 255)
        image[0:len(pixels), 0] = pixels
        histogram = Histogram.create_with_image(image)
        output = histogram.pretty()
    elif transformation_id == 'reverse':
        pixels.reverse()
        output = rle_serialize_line_inner(pixels)
    elif transformation_id == 'compress':
        pixels = list_compress(pixels)
        output = rle_serialize_line_inner(pixels)
    elif transformation_id == 'scaleup':
        pixels = list_scaleup(pixels, scaleup_factor)
        output = rle_serialize_line_inner(pixels)
    else:
        raise Exception("Unreachable code reached")

    benchmark_length = image_size1d_to_string(pixel_length)
    benchmark_id = f'dataset=rle_deserialize group={transformation_id} pixel_length={benchmark_length}'

    result_dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

class DatasetRLE(DatasetGenerator):
    def generate_dataset_item_list(self, seed: int, show: bool) -> list[dict]:
        if seed % 6 == 0:
            item = generate_serialize_dataset_item(seed)
        else:
            item = generate_deserialize_dataset_item(seed)
        return [item]

if __name__ == "__main__":
    generator = DatasetRLE()
    generator.generate(
        seed=1200001,
        max_num_samples=1000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    # generator.inspect()
