# Translate the image by 1 pixel, up/down/left/right.
# Determine what kind of translation happened.
#
# It would be easier if I create an entire ARC task json file.
# On the other hand, I need the benchmark ids, so I can determine what the model struggles with.
#
# Present the same input images, but with different transformations.
# so there are examples of up, down, left, right, and the model should determine what happened.
import random
import os
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.benchmark import *

DATASET_NAMES = [
    'SIMONARCTASKTRANSLATE',
    'SIMONSARCTASKTRANSLATE',
    'SIMONTASKTRANSLATE',
    'SIMONSTASKTRANSLATE',
    'Simon-Task-Translate',
    'Simons-Task-Translate',
    'simon-task-translate',
    'simons-task-translate'
]

def generate_task(seed: int, dx: int, dy: int, percent_noise: float) -> Task:
    count_example = random.Random(seed + 1).randint(2, 3)
    # count_test = random.Random(seed + 2).randint(1, 3)
    count_test = 1
    task = Task()
    min_width = 1
    max_width = 4
    min_height = 1
    max_height = 4

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

def generate_dataset_item_for_output_row(seed: int, task: Task, test_index: int, test_output_y: int, pixel_list: list[int], transformation_id: str) -> dict:
    dataset_name = random.choice(DATASET_NAMES)

    instruction = f"{dataset_name}, test {test_index}, predict row {test_output_y} from the output image"

    task_formatter = TaskFormatterRLE(task)
    input = task_formatter.to_string()
    # print(input)

    output = ''.join(map(str, pixel_list))

    max_width, max_height = task.max_image_size()
    benchmark_width = image_size1d_to_string(max_width)
    benchmark_height = image_size1d_to_string(max_height)
    benchmark_pixels = task_pixels_to_string(task.total_pixel_count())
    benchmark_id = f'dataset=task_solve group={transformation_id} image_width={benchmark_width} image_height={benchmark_height} task_pixels={benchmark_pixels}'

    result_dict = {
        'instruction': instruction,
        'input': input,
        'output': output,
        'benchmark': benchmark_id
    }
    return result_dict

def generate_dataset_item_list(seed):
    random.seed(seed)

    directions = [
        (0, 1, 'translate_yplus1'), 
        (0, -1, 'translate_yminus1'), 
        (1, 0, 'translate_xplus1'), 
        (-1, 0, 'translate_xminus1')
    ]
    dx, dy, transformation_id = random.choice(directions)
    percent_noise = 0.0
    task = generate_task(seed + 1, dx, dy, percent_noise)
    # task.show()

    task_without_test_output = task.clone()
    for test_index in range(task.count_tests):
        task_without_test_output.output_images[task.count_examples + test_index] = None

    dataset_items = []
    for test_index in range(task.count_tests):
        # input_image = task.test_input(test_index)
        output_image = task.test_output(test_index)
        # print(f"Test {test_index}")
        # print(input_image)
        # print(output_image)

        output_height = output_image.shape[0]
        for output_y in range(output_height):
            pixels = image_get_row_as_list(output_image, output_y)
            # print(pixels)

            dataset_item = generate_dataset_item_for_output_row(seed, task_without_test_output, test_index, output_y, pixels, transformation_id)
            # print(dataset_item)
            dataset_items.append(dataset_item)

    return dataset_items

#generate_dataset_item_batch(0)
#exit()

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
    max_num_samples=100,
    max_byte_size=1024*1024*100,
)

# Save dataset to file
filename = 'dataset_task_solve.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

