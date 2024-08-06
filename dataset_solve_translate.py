# Translate the image by 1 pixel, up/down/left/right.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import random
import os
from tqdm import tqdm
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
from dataset.generate_solve import *

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

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_height()
    # builder.append_pixels()
    builder.append_image()
    return builder.dataset_items()

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

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=2400000):
    dataset = []
    dataset_byte_size = 0
    stop = False
    with tqdm(total=max_num_samples, desc="Generating dataset", ncols=100, unit="sample", mininterval=1.0, dynamic_ncols=True, leave=True) as pbar:
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
                pbar.update(1)
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

