# color transformations:
# - Swap 2 colors.
# - Identify the most popular color.
# - Identify the least popular color.
#
# IDEA: Replace one color with another color.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import random
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_verbose import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from dataset.generate_solve import *
from dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_color'
SAVE_FILENAME = 'dataset_solve_color.jsonl'

def generate_task_swap_colors(seed: int) -> Task:
    random.seed(seed)

    count_example = random.randint(2, 4)
    count_test = random.randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 1
    max_width = 14
    min_height = 1
    max_height = 14

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
            width = random.randint(use_min_width, max_width)
            height = random.randint(use_min_height, max_height)
            ratios = [0.2, 0.3, 0.4, 0.5]
            ratio = random.choice(ratios)
            mask_image = image_create_random_with_two_colors(width, height, 0, 1, ratio, seed + 1060 + i)
            histogram = Histogram.create_with_image(mask_image)
            if histogram.number_of_unique_colors() == 2:
                # print(f"retry_index: {retry_index}")
                break

        if mask_image is None:
            raise ValueError(f"Failed to create mask_image with 2 colors")

        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.shuffle(colors)
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

def generate_task_mostleast_popular_color(seed: int, find_id: str, output_size_id: str) -> Task:
    random.seed(seed)

    count_example = random.randint(2, 4)
    count_test = random.randint(1, 2)
    # count_test = 1
    task = Task()
    min_width = 1
    max_width = 14
    min_height = 1
    max_height = 14

    for i in range(count_example+count_test):
        is_example = i < count_example

        random_image = None
        found_color = None
        number_of_retries = 0
        for retry_index in range(100):
            use_min_width = min_width
            use_min_height = min_height
            if retry_index == 1:
                use_min_width = 2
                use_min_height = 2
            if retry_index >= 2:
                use_min_width = 3
                use_min_height = 3
            random_image = image_create_random_advanced(seed + i + retry_index, use_min_width, max_width, use_min_height, max_height)
            histogram = Histogram.create_with_image(random_image)
            found_color = None
            if find_id == 'most_popular':
                found_color = histogram.most_popular_color()
            elif find_id == 'least_popular':
                found_color = histogram.least_popular_color()
            else:
                raise ValueError(f"Unknown find_id: {find_id}")
            
            if found_color is not None:
                number_of_retries = retry_index
                break

        if random_image is None:
            raise ValueError(f"Failed to create random image")
        if found_color is None:
            raise ValueError(f"Failed to find color")
        if number_of_retries >= 50:
            print(f"number_of_retries: {number_of_retries}")

        input_image = random_image

        output_width = None
        output_height = None
        if output_size_id == '1x1':
            output_width = 1
            output_height = 1
        elif output_size_id == 'same':
            output_width = input_image.shape[1]
            output_height = input_image.shape[0]
        else:
            raise ValueError(f"Unknown output_size_id: {output_size_id}")
        
        output_image = image_create(output_width, output_height, found_color)

        task.append_pair(input_image, output_image, is_example)

    return task

def demo_generate_task():
    for i in range(10):
        # task = generate_task_swap_colors(i)
        # task = generate_task_mostleast_popular_color(i, 'most_popular', '1x1')
        task = generate_task_mostleast_popular_color(i, 'least_popular', '1x1')
        # task = generate_task_mostleast_popular_color(i, 'most_popular', 'same')
        # task = generate_task_mostleast_popular_color(i, 'least_popular', 'same')
        task.show()

# demo_generate_task()
# exit()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    # builder.append_height()
    # builder.append_pixels()
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    random.seed(seed)

    seed_task = seed

    transformation_ids = [
        'swap_colors',
        'most_popular_color_1x1',
        'least_popular_color_1x1',
        'most_popular_color_same',
        'least_popular_color_same',
    ]

    all_dataset_items = []
    for transformation_id in transformation_ids:
        task = None
        if transformation_id == 'swap_colors':
            task = generate_task_swap_colors(seed_task)
        elif transformation_id == 'most_popular_color_1x1':
            task = generate_task_mostleast_popular_color(seed_task, 'most_popular', '1x1')
        elif transformation_id == 'least_popular_color_1x1':
            task = generate_task_mostleast_popular_color(seed_task, 'least_popular', '1x1')
        elif transformation_id == 'most_popular_color_same':
            task = generate_task_mostleast_popular_color(seed_task, 'most_popular', 'same')
        elif transformation_id == 'least_popular_color_same':
            task = generate_task_mostleast_popular_color(seed_task, 'least_popular', 'same')
        else:
            raise ValueError(f"Unknown transformation_id: {transformation_id}")
        
        # task.show()
        dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
        all_dataset_items.extend(dataset_items)

    return all_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=8600232,
    max_num_samples=100000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILENAME)
