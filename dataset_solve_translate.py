# Translate the image by 1 pixel, up/down/left/right.
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
from simon_arc_lab.benchmark import *
from dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from dataset.plot import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES

BENCHMARK_DATASET_NAME = 'solve_translate'

def generate_task(seed: int, dx: int, dy: int, percent_noise: float) -> Task:
    count_example = random.Random(seed + 1).randint(2, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 3
    max_width = 9
    min_height = 3
    max_height = 9

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1000 + i, min_width, max_width, min_height, max_height)

        transformed_image = image_translate_wrap(input_image, dx, dy)

        height, width = transformed_image.shape
        noise_image = image_create_random_advanced(seed + 1001 + i, width, width, height, height)
        mask = image_create_random_with_two_colors(width, height, 0, 1, percent_noise, seed + 1050 + i)

        output_image = image_mix(mask, transformed_image, noise_image)

        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    ratios = [0.0, 0.33, 0.5]
    for i in range(3):
        ratio = ratios[i]
        task = generate_task(0, 0, 1, ratio)
        task.show()

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

    directions = [
        (1, 0, 'translate_xplus1'), 
        (-1, 0, 'translate_xminus1'),
        (0, 1, 'translate_yplus1'), 
        (0, -1, 'translate_yminus1'),
        (1, -1, 'translate_xplus1yminus1'), 
        (-1, -1, 'translate_xminus1minus1'),
        (1, 1, 'translate_xplus1yplus1'), 
        (-1, 1, 'translate_xminus1plus1'),
    ]

    all_dataset_items = []
    for direction in directions:
        dx, dy, transformation_id = direction
        percent_noise = 0.0
        task = generate_task(seed_task, dx, dy, percent_noise)
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=2200000):
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
    max_num_samples=100000,
    max_byte_size=1024*1024*100,
)

# plot_prompt_length_distribution(dataset)
# plot_response_length_distribution(dataset)

# Save dataset to file
filename = 'dataset_solve_translate.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

