# Replace one color with another color.
# Swap 2 colors.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import random
import os
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *

DATASET_NAMES = [
    'SIMONARCSOLVECOLOR',
    'SIMONSARCSOLVECOLOR',
    'SIMONSOLVECOLOR',
    'SIMONSSOLVECOLOR',
    'Simon-Solve-Color',
    'Simons-Solve-Color',
    'simon-solve-color',
    'simons-solve-color'
]

BENCHMARK_DATASET_NAME = 'solve_color'

def generate_task_swap_colors(seed: int) -> Task:
    count_example = random.Random(seed + 9).randint(2, 3)
    # count_test = random.Random(seed + 10).randint(1, 2)
    count_test = 1
    task = Task()
    min_width = 1
    max_width = 5
    min_height = 1
    max_height = 5

    for i in range(count_example+count_test):
        is_example = i < count_example

        mask_image = None
        for retry_index in range(10):
            use_min_width = min_width
            use_min_height = min_height
            if retry_index == 1:
                use_min_width = 2
                use_min_height = 2
            if retry_index >= 2:
                use_min_width = 3
                use_min_height = 3
            width = random.Random(seed + 1 + i).randint(use_min_width, max_width)
            height = random.Random(seed + 2 + i).randint(use_min_height, max_height)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.Random(seed + 3 + i).choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, seed + 1060 + i)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() == 2:
                print(f"retry_index: {retry_index}")
                break

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")

        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(seed + 4 + i).shuffle(colors)
        color0 = colors[0]
        color1 = colors[1]

        color_map = {
            0: color0,
            1: color1,
        }
        color_map_swapped = {
            0: color1,
            1: color0,
        }

        input_image = image_replace_colors(mask_image, color_map)
        output_image = image_replace_colors(mask_image, color_map_swapped)

        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(10):
        task = generate_task_swap_colors(i + 20)
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_for_pixels_in_output_row(seed: int, task: Task, test_index: int, test_output_y: int, pixel_list: list[int], transformation_id: str) -> dict:
    random.seed(seed)
    dataset_name = random.choice(DATASET_NAMES)

    # task_formatter = TaskFormatterRLEVerbose(task)
    task_formatter = TaskFormatterRLECompact(task)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict row {test_output_y}",
        f"{dataset_name} '{test_output_id}' predict row {test_output_y}",
        f"{dataset_name} '{test_output_id}' predict the row {test_output_y}",
        f"{dataset_name}, '{test_output_id}', predict the row {test_output_y}",
        f"{dataset_name}, '{test_output_id}', predict y={test_output_y}",
        f"{dataset_name} {test_output_id} predict y={test_output_y}",
        f"{dataset_name} predict y={test_output_y} for {test_output_id}",
        f"{dataset_name} predict row {test_output_y} for {test_output_id}",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output = ''.join(map(str, pixel_list))

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={BENCHMARK_DATASET_NAME} group={transformation_id} predict=pixels image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_for_number_of_output_rows(seed: int, task: Task, test_index: int, output_image: np.array, transformation_id: str) -> dict:
    random.seed(seed)
    dataset_name = random.choice(DATASET_NAMES)

    # task_formatter = TaskFormatterRLEVerbose(task)
    task_formatter = TaskFormatterRLECompact(task)

    output_ids = task_formatter.output_ids()
    test_output_id = output_ids[task.count_examples + test_index]

    instructions = [
        f"{dataset_name}, {test_output_id}, predict row count",
        f"{dataset_name} '{test_output_id}' predict row count",
        f"{dataset_name} '{test_output_id}' predict the row count",
        f"{dataset_name}, '{test_output_id}', predict the row count",
        f"{dataset_name}, '{test_output_id}', predict the height",
        f"{dataset_name}, '{test_output_id}', predict height",
        f"{dataset_name} {test_output_id} predict the height",
        f"{dataset_name} {test_output_id} predict height",
    ]
    instruction = random.choice(instructions)

    input = task_formatter.to_string()
    # print(input)

    output_height = output_image.shape[0]
    output = str(output_height)
    # print(output)

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset={BENCHMARK_DATASET_NAME} group={transformation_id} predict=height image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    random.seed(seed)

    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()

    dataset_items = []

    # Predict the pixels of the output image
    for test_index in range(task.count_tests):
        output_image = task.test_output(test_index)
        output_height = output_image.shape[0]
        for output_y in range(output_height):
            pixels = image_get_row_as_list(output_image, output_y)
            dataset_item = generate_dataset_item_for_pixels_in_output_row(seed + output_y + test_index * 100, task_without_test_output, test_index, output_y, pixels, transformation_id)
            dataset_items.append(dataset_item)

    # Predict the number of rows in the output image
    for test_index in range(task.count_tests):
        output_image = task.test_output(test_index)
        dataset_item = generate_dataset_item_for_number_of_output_rows(seed + test_index * 100 + 1000, task_without_test_output, test_index, output_image, transformation_id)
        dataset_items.append(dataset_item)

    return dataset_items

def generate_dataset_item_list(seed: int) -> list[dict]:
    random.seed(seed)

    seed_task = seed

    transformation_ids = [
        'swap_colors',
    ]

    all_dataset_items = []
    for transformation_id in transformation_ids:
        task = None
        if transformation_id == 'swap_colors':
            task = generate_task_swap_colors(seed_task)
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")
        
        task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=100000):
    dataset = []
    dataset_byte_size = 0
    stop = False
    for i in range(max_num_samples):
        if stop:
            break
        items = generate_dataset_item_list(seed_start + i)
        for item in items:
            bytes = len(json.dumps(item))
            if dataset_byte_size + bytes > max_byte_size:
                stop = True
                break
            if len(dataset) >= max_num_samples:
                stop = True
                break
            dataset_byte_size += bytes
            dataset.append(item)
    random.Random(seed_start).shuffle(dataset)
    return dataset

dataset = generate_dataset(
    max_num_samples=10,
    max_byte_size=1024*1024*100,
)

# Save dataset to file
filename = 'dataset_solve_color.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

