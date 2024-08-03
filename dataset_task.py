# IDEA: Currently it's image sizes 1-5 that are being exercised. Extend this to bigger sizes.
# IDEA: extract the image pair number N, that contains both input+output
# IDEA: comparison of 2 images, add histograms
# IDEA: comparison of 2 images, subtract histograms
# IDEA: comparison of 2 images, are they the same, are they different
# IDEA: extract output A
# IDEA: histogram of input B
# IDEA: rotate cw output D
# IDEA: flipx input E
# IDEA: compare input E and output E
# IDEA: intersection of all input histograms
# IDEA: union of all input histograms
# IDEA: intersection of all output histograms
# IDEA: are the sizes of the input and output the same?
# IDEA: union of all output histograms
import json
import os
import random
from simon_arc_lab.rle.serialize import serialize
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *

def generate_task(seed):
    count_example = random.Random(seed + 1).randint(2, 5)
    count_test = random.Random(seed + 2).randint(1, 3)
    task = Task()
    min_width = 1
    max_width = 5
    min_height = 1
    max_height = 5

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1000 + i, min_width, max_width, min_height, max_height)
        if is_example:
            output_image = image_create_random_advanced(seed + 2000 + i, min_width, max_width, min_height, max_height)
        else:
            output_image = None
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item(seed):
    dataformat_names = [
        'SIMONARCTASK',
        'SIMONSARCTASK',
        'simonarctask',
        'simons-arc-task',
        'Simon-ARC-Task',
        'SimonsArcTask',
        'SimonsArcRLETask',
        'Simons-Arc-RLE-Task',
        'simon-arc-rle-task',
        'simons-arc-rle-task',
    ]
    dataformat_name = random.Random(seed + 1004).choice(dataformat_names)

    transformation_ids = [
        'extract_input_by_id', 
        'extract_output_by_id', 
        'histogram_input_by_id', 
        'histogram_output_by_id',
        'flipx_input_by_id',
        'flipx_output_by_id',
        'flipy_input_by_id',
        'flipy_output_by_id',
        'pair_histogram_intersection_colors',
        'pair_histogram_union_colors',
        'pair_histogram_add',
        'pair_histogram_max',
        'pair_histogram_min',
    ]
    transformation_weights = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # transformation_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50]
    # transformation_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0]
    transformation_id = random.Random(seed + 1001).choices(transformation_ids, weights=transformation_weights, k=1)[0]


    task = generate_task(seed + 1002)

    # task_formatter = TaskFormatterRLEVerbose(task)
    task_formatter = TaskFormatterRLECompact(task)
    input = task_formatter.to_string()

    output = None
    instruction = None

    if transformation_id == 'extract_input_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.input_ids()[pair_index]
        output = task_formatter.serialize_input_image(pair_index)
        instructions = [
            f"This is {dataformat_name} data. Extract {image_id}",
            f"This is {dataformat_name} data. Extract '{image_id}'",
            f"{dataformat_name}, return {image_id}",
            f"{dataformat_name}, return '{image_id}'",
            f"{dataformat_name}, get image {image_id}",
            f"{dataformat_name}, get image '{image_id}'",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'extract_output_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.output_ids()[pair_index]
        output = task_formatter.serialize_output_image(pair_index)
        instructions = [
            f"This is {dataformat_name} data. Extract {image_id}",
            f"This is {dataformat_name} data. Extract '{image_id}'",
            f"{dataformat_name}, return {image_id}",
            f"{dataformat_name}, return '{image_id}'",
            f"{dataformat_name}, get image {image_id}",
            f"{dataformat_name}, get image '{image_id}'",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'histogram_input_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.input_ids()[pair_index]
        histogram = Histogram.create_with_image(task.input_images[pair_index])
        output = histogram.pretty()
        instructions = [
            f"This is {dataformat_name} data. Histogram of {image_id}",
            f"This is {dataformat_name} data. Histogram of '{image_id}'",
            f"{dataformat_name}, return histogram of {image_id}",
            f"{dataformat_name}, return histogram of '{image_id}'",
            f"{dataformat_name}, get histogram for {image_id}",
            f"{dataformat_name}, get histogram for '{image_id}'",
            f"{dataformat_name}, process {image_id} and return histogram",
            f"{dataformat_name}, process '{image_id}' and return histogram",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'histogram_output_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.output_ids()[pair_index]
        image = task.output_images[pair_index]
        if image is None:
            output = "None"
        else:
            histogram = Histogram.create_with_image(image)
            output = histogram.pretty()
        instructions = [
            f"This is {dataformat_name} data. Histogram of {image_id}",
            f"This is {dataformat_name} data. Histogram of '{image_id}'",
            f"{dataformat_name}, return histogram of {image_id}",
            f"{dataformat_name}, return histogram of '{image_id}'",
            f"{dataformat_name}, get histogram for {image_id}",
            f"{dataformat_name}, get histogram for '{image_id}'",
            f"{dataformat_name}, process {image_id} and return histogram",
            f"{dataformat_name}, process '{image_id}' and return histogram",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'flipx_input_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.input_ids()[pair_index]
        image = task.input_images[pair_index]
        flipped_image = image[:, ::-1]
        output = serialize(flipped_image)
        instructions = [
            f"This is {dataformat_name} data. FlipX {image_id}",
            f"This is {dataformat_name} data. Flip-X '{image_id}'",
            f"{dataformat_name}, return flipx of {image_id}",
            f"{dataformat_name}, return flip-x of '{image_id}'",
            f"{dataformat_name}, get {image_id} and flipx",
            f"{dataformat_name}, get {image_id} and Flip-X",
            f"{dataformat_name}, get {image_id} and FlipX",
            f"{dataformat_name}, process {image_id} and return flipx",
            f"{dataformat_name}, process '{image_id}' and return Flip-X",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'flipx_output_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.output_ids()[pair_index]
        image = task.output_images[pair_index]
        if image is None:
            output = "None"
        else:
            flipped_image = image[:, ::-1]
            output = serialize(flipped_image)
        instructions = [
            f"This is {dataformat_name} data. FlipX {image_id}",
            f"This is {dataformat_name} data. Flip-X '{image_id}'",
            f"{dataformat_name}, return flipx of {image_id}",
            f"{dataformat_name}, return flip-x of '{image_id}'",
            f"{dataformat_name}, get {image_id} and flipx",
            f"{dataformat_name}, get {image_id} and Flip-X",
            f"{dataformat_name}, get {image_id} and FlipX",
            f"{dataformat_name}, process {image_id} and return flipx",
            f"{dataformat_name}, process '{image_id}' and return Flip-X",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'flipy_input_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.input_ids()[pair_index]
        image = task.input_images[pair_index]
        flipped_image = image[::-1, :]
        output = serialize(flipped_image)
        instructions = [
            f"This is {dataformat_name} data. FlipY {image_id}",
            f"This is {dataformat_name} data. Flip-Y '{image_id}'",
            f"{dataformat_name}, return flipy of {image_id}",
            f"{dataformat_name}, return flip-y of '{image_id}'",
            f"{dataformat_name}, get {image_id} and flipy",
            f"{dataformat_name}, get {image_id} and Flip-Y",
            f"{dataformat_name}, get {image_id} and FlipY",
            f"{dataformat_name}, process {image_id} and return flipy",
            f"{dataformat_name}, process '{image_id}' and return Flip-Y",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'flipy_output_by_id':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        image_id = task_formatter.output_ids()[pair_index]
        image = task.output_images[pair_index]
        if image is None:
            output = "None"
        else:
            flipped_image = image[::-1, :]
            output = serialize(flipped_image)
        instructions = [
            f"This is {dataformat_name} data. FlipY {image_id}",
            f"This is {dataformat_name} data. Flip-Y '{image_id}'",
            f"{dataformat_name}, return flipy of {image_id}",
            f"{dataformat_name}, return flip-y of '{image_id}'",
            f"{dataformat_name}, get {image_id} and flipy",
            f"{dataformat_name}, get {image_id} and Flip-Y",
            f"{dataformat_name}, get {image_id} and FlipY",
            f"{dataformat_name}, process {image_id} and return flipy",
            f"{dataformat_name}, process '{image_id}' and return Flip-Y",
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'pair_histogram_intersection_colors':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        pair_id = task_formatter.pair_ids()[pair_index]
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        if output_image is None:
            output = "maybe"
        else:
            histogram_input = Histogram.create_with_image(input_image)
            histogram_output = Histogram.create_with_image(output_image)
            output = histogram_input.color_intersection_pretty(histogram_output)
        instructions = [
            f'{dataformat_name}, {pair_id}, unique colors that the images have in common',
            f'{dataformat_name}, {pair_id}, unique colors that the two images have in common',
            f'{dataformat_name}, {pair_id}, unique colors that the 2 images have in common',
            f'{dataformat_name}, {pair_id}, intersection of colors of the two images',
            f'{dataformat_name}, {pair_id}, intersection of colors of the 2 images',
            f'{dataformat_name}, {pair_id}, intersection of colors of the images',
            f'{dataformat_name}, {pair_id}, overlap of colors of the images',
            f'{dataformat_name}, {pair_id}, color intersection',
            f'{dataformat_name}, {pair_id}, color overlap',
            f'Process {dataformat_name} {pair_id}, and return the intersection of colors',
            f'Process {dataformat_name} {pair_id}, and return the overlap of colors',
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'pair_histogram_union_colors':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        pair_id = task_formatter.pair_ids()[pair_index]
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        if output_image is None:
            output = "maybe"
        else:
            histogram_input = Histogram.create_with_image(input_image)
            histogram_output = Histogram.create_with_image(output_image)
            histogram = histogram_input.add(histogram_output)
            unique_colors = histogram.unique_colors_pretty()
            output = unique_colors
        instructions = [
            f'{dataformat_name}, {pair_id}, unique colors that the images have in the union',
            f'{dataformat_name}, {pair_id}, unique colors that the two images have the union',
            f'{dataformat_name}, {pair_id}, unique colors that the 2 images have the union',
            f'{dataformat_name}, {pair_id}, union of colors of the two images',
            f'{dataformat_name}, {pair_id}, union of colors of the 2 images',
            f'{dataformat_name}, {pair_id}, union of colors of the images',
            f'{dataformat_name}, {pair_id}, union of colors of the images',
            f'{dataformat_name}, {pair_id}, color union',
            f'Process {dataformat_name} {pair_id}, and return the union of colors',
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'pair_histogram_add':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        pair_id = task_formatter.pair_ids()[pair_index]
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        if output_image is None:
            output = "maybe"
        else:
            histogram_input = Histogram.create_with_image(input_image)
            histogram_output = Histogram.create_with_image(output_image)
            histogram = histogram_input.add(histogram_output)
            unique_colors = histogram.pretty()
            output = unique_colors
        instructions = [
            f'{dataformat_name}, {pair_id}, add histograms of the images',
            f'{dataformat_name}, {pair_id}, add histograms of the two images',
            f'{dataformat_name}, {pair_id}, add histograms of the 2 images',
            f'{dataformat_name}, {pair_id}, add histograms',
            f'{dataformat_name}, {pair_id}, the histograms added',
            f'{dataformat_name}, {pair_id}, sum of histograms',
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'pair_histogram_max':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        pair_id = task_formatter.pair_ids()[pair_index]
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        if output_image is None:
            output = "maybe"
        else:
            histogram_input = Histogram.create_with_image(input_image)
            histogram_output = Histogram.create_with_image(output_image)
            histogram = histogram_input.max(histogram_output)
            unique_colors = histogram.pretty()
            output = unique_colors
        instructions = [
            f'{dataformat_name}, {pair_id}, max histogram of the images',
            f'{dataformat_name}, {pair_id}, max histogram of the two images',
            f'{dataformat_name}, {pair_id}, max histogram of the 2 images',
            f'{dataformat_name}, {pair_id}, max histograms',
            f'{dataformat_name}, {pair_id}, the histograms max',
            f'{dataformat_name}, {pair_id}, max of histograms',
            f'{dataformat_name}, {pair_id}, maximum of histograms',
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    if transformation_id == 'pair_histogram_min':
        count = task.count()
        pair_index = random.Random(seed + 1).randint(0, count-1)
        pair_id = task_formatter.pair_ids()[pair_index]
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        if output_image is None:
            output = "maybe"
        else:
            histogram_input = Histogram.create_with_image(input_image)
            histogram_output = Histogram.create_with_image(output_image)
            histogram = histogram_input.min(histogram_output)
            unique_colors = histogram.pretty()
            output = unique_colors
        instructions = [
            f'{dataformat_name}, {pair_id}, min histogram of the images',
            f'{dataformat_name}, {pair_id}, min histogram of the two images',
            f'{dataformat_name}, {pair_id}, min histogram of the 2 images',
            f'{dataformat_name}, {pair_id}, min histograms',
            f'{dataformat_name}, {pair_id}, the histograms min',
            f'{dataformat_name}, {pair_id}, min of histograms',
            f'{dataformat_name}, {pair_id}, minimum of histograms',
        ]
        instruction = random.Random(seed + 1006).choice(instructions)

    debug = False
    if debug:
        print("---")
        print("instruction:")
        print(instruction)
        print("input:")
        print(input)
        print("output:")
        print(output)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset=task group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=1200000):
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
    max_num_samples=100000,
    max_byte_size=1024*1024*100,
)

# Save dataset to file
filename = 'dataset_task.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

