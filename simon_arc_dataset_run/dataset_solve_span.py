# Span intersections.
#
# Present the same input images, but with different transformations.
# so from the examples alone, the model have to determine what happened.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import random
from simon_arc_lab.image_mix import *
from simon_arc_lab.image_util import *
from simon_arc_lab.image_create_random_advanced import image_create_random_advanced
from simon_arc_lab.task import *
from simon_arc_lab.image_create_random_simple import *
from simon_arc_lab.image_mix import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_lab.image_rect import image_rect_inside
from simon_arc_lab.image_scale import image_scale_up_variable
from simon_arc_lab.rectangle import Rectangle
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_span'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_span.jsonl')

def grid2x2(topleft: np.array, topright: np.array, bottomleft: np.array, bottomright: np.array) -> np.array:
    row_top = np.hstack([topleft, topright])
    row_bottom = np.hstack([bottomleft, bottomright])
    return np.vstack([row_top, row_bottom])

def generate_task_with_intersecting_spans(seed: int, transformation_id: str) -> Task:
    """
    Spans that are intersecting.

    Examples:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=a406ac07
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_span_count = 3
    max_span_count = 6
    max_image_size = 8

    color_background = 9
    color_template = 8

    span_color_count = random.Random(seed + 3).randint(2, 6)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(colors)
    color_mapping = {}
    for i in range(10):
        color_mapping[i] = colors[i]

    task.metadata_task_id = f'intersecting_spans {transformation_id}'

    # Rotate the images
    rotate90_count = random.Random(seed + 3).randint(0, 3)

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            iteration_seed = seed * 5 + retry_index * 133 + i * 1000

            span_count = random.Random(iteration_seed + 1).randint(min_span_count, max_span_count)

            image_width = random.Random(iteration_seed + 2).randint(span_count, max_image_size)
            image_height = random.Random(iteration_seed + 3).randint(span_count, max_image_size)

            x_span_list = [1] * span_count
            for i in range(image_width - span_count):
                item_index = random.Random(iteration_seed + 4 + i * 10000).randint(0, span_count - 1)
                x_span_list[item_index] += 1

            y_span_list = [1] * span_count
            for i in range(image_height - span_count):
                item_index = random.Random(iteration_seed + 5 + i * 383838).randint(0, span_count - 1)
                y_span_list[item_index] += 1

            span_color_list = []
            last_color = None
            for i in range(span_count):
                color = random.Random(iteration_seed + 6 + i).randint(0, span_color_count - 1)
                if color == last_color:
                    color = (last_color + 1) % span_color_count
                last_color = color
                span_color_list.append(color)

            # Primary area
            primary_empty = image_create(image_width, image_height, color_background)
            primary_color_image = image_create(image_width, image_height, color_background)
            primary_mask = image_create(image_width, image_height, color_background)
            current_y = 0
            for span_y in range(span_count):
                height = y_span_list[span_y]
                y = current_y
                current_y += height
                current_x = 0
                for span_x in range(span_count):
                    width = x_span_list[span_x]
                    x = current_x
                    current_x += width

                    if span_x == span_y:
                        color = span_color_list[span_x]
                        primary_color_image = image_rect_inside(primary_color_image, Rectangle(x, y, width, height), color)
                        primary_mask = image_rect_inside(primary_mask, Rectangle(x, y, width, height), color_template)

            # Vertical border
            vertical_image = image_create(1, image_height, color_background)
            current_y = 0
            for span_y in range(span_count):
                color = span_color_list[span_y]
                height = y_span_list[span_y]
                y = current_y
                current_y += height
                vertical_image = image_rect_inside(vertical_image, Rectangle(0, y, 1, height), color)

            # Horizontal border
            horizontal_image = image_create(image_width, 1, color_background)
            current_x = 0
            for span_x in range(span_count):
                color = span_color_list[span_x]
                width = x_span_list[span_x]
                x = current_x
                current_x += width
                horizontal_image = image_rect_inside(horizontal_image, Rectangle(x, 0, width, 1), color)

            color_of_last_span = span_color_list[-1]
            bottom_right_image = image_create(1, 1, color_of_last_span)

            if transformation_id == 'empty_primary_area':
                input_image_raw = grid2x2(primary_empty, vertical_image, horizontal_image, bottom_right_image)
                output_image_raw = grid2x2(primary_color_image, vertical_image, horizontal_image, bottom_right_image)
            elif transformation_id == 'template_primary_area':
                input_image_raw = grid2x2(primary_mask, vertical_image, horizontal_image, bottom_right_image)
                output_image_raw = grid2x2(primary_color_image, vertical_image, horizontal_image, bottom_right_image)
            elif transformation_id == 'template_primary_area_without_border':
                input_image_raw = grid2x2(primary_mask, vertical_image, horizontal_image, bottom_right_image)
                output_image_raw = primary_color_image
            elif transformation_id == 'colored_primary_area_fill_template_border':
                input_image_raw = primary_color_image
                output_image_raw = grid2x2(primary_color_image, vertical_image, horizontal_image, bottom_right_image)
            elif transformation_id == 'colored_primary_area_extract_horizontal':
                input_image_raw = primary_color_image
                output_image_raw = horizontal_image
            elif transformation_id == 'colored_primary_area_extract_vertical':
                input_image_raw = primary_color_image
                output_image_raw = vertical_image
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            # Rotate
            input_image_raw = np.rot90(input_image_raw, rotate90_count)
            output_image_raw = np.rot90(output_image_raw, rotate90_count)

            # Palette
            input_image = image_replace_colors(input_image_raw, color_mapping)
            output_image = image_replace_colors(output_image_raw, color_mapping)

            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_template_lines(seed: int, transformation_id: str) -> Task:
    """
    Lines that are obscured by a mask. Fill in the template.

    Examples:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c7d4e6ad
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c9f8e694
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_span_count = 3
    max_span_count = 6
    max_image_size = 8

    color_background = 9
    color_template = 8

    span_color_count = random.Random(seed + 3).randint(2, 6)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(colors)
    color_mapping = {}
    for i in range(10):
        color_mapping[i] = colors[i]

    color_mapping_mask_to_template = {
        0: color_background,
        1: color_template,
    }

    task.metadata_task_id = f'template_lines {transformation_id}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            iteration_seed = seed * 5 + retry_index * 133 + i * 1000

            span_count = random.Random(iteration_seed + 1).randint(min_span_count, max_span_count)

            image_width = random.Random(iteration_seed + 2).randint(span_count, max_image_size)
            image_height = random.Random(iteration_seed + 3).randint(span_count, max_image_size)

            ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ratio = random.Random(iteration_seed + 11).choice(ratios)
            random_image = image_create_random_with_two_colors(image_width, image_height, 0, 1, ratio, iteration_seed + 12)
            histogram = Histogram.create_with_image(random_image)
            if histogram.number_of_unique_colors() < 2:
                continue

            span_list = [1] * span_count
            for i in range(image_width - span_count):
                item_index = random.Random(iteration_seed + 5 + i * 383838).randint(0, span_count - 1)
                span_list[item_index] += 1

            span_color_list = []
            last_color = None
            for i in range(span_count):
                color = random.Random(iteration_seed + 6 + i).randint(0, span_color_count - 1)
                if color == last_color:
                    color = (last_color + 1) % span_color_count
                last_color = color
                span_color_list.append(color)

            # Horizontal border
            horizontal_image = image_create(image_width, 1, color_background)
            current_x = 0
            for span_x in range(span_count):
                color = span_color_list[span_x]
                width = span_list[span_x]
                x = current_x
                current_x += width
                horizontal_image = image_rect_inside(horizontal_image, Rectangle(x, 0, width, 1), color)

            # repeat the horizontal image by the height
            primary_image = np.tile(horizontal_image, (image_height, 1))
            primary_empty = image_create(image_width, image_height, color_background)
            primary_image = image_mix(random_image, primary_image, primary_empty)

            template_image = image_replace_colors(random_image, color_mapping_mask_to_template)
            if transformation_id == 'output_with_border':
                input_image_raw = np.vstack([horizontal_image, template_image])
                output_image_raw = np.vstack([horizontal_image, primary_image])
            elif transformation_id == 'output_without_border':
                input_image_raw = np.vstack([horizontal_image, template_image])
                output_image_raw = primary_image
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            # Rotate
            rotate90_count = random.Random(iteration_seed + 30).randint(0, 3)
            input_image_raw = np.rot90(input_image_raw, rotate90_count)
            output_image_raw = np.rot90(output_image_raw, rotate90_count)

            # Palette
            input_image = image_replace_colors(input_image_raw, color_mapping)
            output_image = image_replace_colors(output_image_raw, color_mapping)

            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_task_with_alternate(seed: int, transformation_id: str) -> Task:
    """
    Alternating lines that are intersecting with operations: and, or, xor, sum.

    Examples:
    https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2281f1f4
    """
    count_example = random.Random(seed + 1).randint(2, 3)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_span_count = 4
    max_span_count = 5
    max_image_size = 8

    color_background = 9
    color_indicator = 8

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 5).shuffle(input_colors)
    input_color_map = {}
    for i in range(10):
        input_color_map[i] = input_colors[i]

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 6).shuffle(output_colors)
    output_color_map = {}
    for i in range(10):
        output_color_map[i] = output_colors[i]

    set_color0_to_background = random.Random(seed + 7).randint(0, 1)
    set_color1_to_indicator = random.Random(seed + 8).randint(0, 1)
    color_map_primary = {}
    if set_color0_to_background > 0:
        color_map_primary[0] = color_background
    if set_color1_to_indicator > 0:
        color_map_primary[1] = color_indicator

    has_output_with_border = random.Random(seed + 9).choice([False, True])

    task.metadata_task_id = f'alternating_intersection {transformation_id} outputborder={has_output_with_border}'

    for i in range(count_example+count_test):
        is_example = i < count_example
        output_image = None
        input_image = None
        for retry_index in range(100):
            iteration_seed = seed * 5 + retry_index * 133 + i * 1000

            x_span_count = random.Random(iteration_seed + 1).randint(min_span_count, max_span_count)
            y_span_count = random.Random(iteration_seed + 2).randint(min_span_count, max_span_count)

            image_width = random.Random(iteration_seed + 3).randint(x_span_count, max_image_size)
            image_height = random.Random(iteration_seed + 4).randint(y_span_count, max_image_size)

            x_span_list = [1] * x_span_count
            for i in range(image_width - x_span_count):
                item_index = random.Random(iteration_seed + 5 + i * 10000).randint(0, x_span_count - 1)
                x_span_list[item_index] += 1

            y_span_list = [1] * y_span_count
            for i in range(image_height - y_span_count):
                item_index = random.Random(iteration_seed + 6 + i * 383838).randint(0, y_span_count - 1)
                y_span_list[item_index] += 1

            # Alternating values
            x_span_value_list = []
            x_value = random.Random(iteration_seed + 7 + i).randint(0, 1)
            for i in range(x_span_count):
                value = (x_value + 1) % 2
                x_value = value
                x_span_value_list.append(value)

            # Alternating values
            y_span_value_list = []
            y_value = random.Random(iteration_seed + 8 + i).randint(0, 1)
            for i in range(y_span_count):
                value = (y_value + 1) % 2
                y_value = value
                y_span_value_list.append(value)

            primary_empty = image_create(image_width, image_height, color_background)

            # Primary area
            image_and = image_create(image_width, image_height, 0)
            image_or = image_create(image_width, image_height, 0)
            image_xor = image_create(image_width, image_height, 0)
            image_sum = image_create(image_width, image_height, 0)
            current_y = 0
            for span_y in range(y_span_count):
                height = y_span_list[span_y]
                y = current_y
                current_y += height
                current_x = 0
                for span_x in range(x_span_count):
                    width = x_span_list[span_x]
                    x = current_x
                    current_x += width

                    rect = Rectangle(x, y, width, height)
                    value0 = x_span_value_list[span_x]
                    value1 = y_span_value_list[span_y]
                    image_and = image_rect_inside(image_and, rect, value0 & value1)
                    image_or  = image_rect_inside(image_or,  rect, value0 | value1)
                    image_xor = image_rect_inside(image_xor, rect, value0 ^ value1)
                    image_sum = image_rect_inside(image_sum, rect, value0 + value1)

            # Vertical border indicators
            vertical_image = image_create(1, image_height, color_background)
            current_y = 0
            for span_y in range(y_span_count):
                value = y_span_value_list[span_y]
                height = y_span_list[span_y]
                y = current_y
                current_y += height
                if value > 0:
                    vertical_image = image_rect_inside(vertical_image, Rectangle(0, y, 1, height), color_indicator)

            # Horizontal border indicators
            horizontal_image = image_create(image_width, 1, color_background)
            current_x = 0
            for span_x in range(x_span_count):
                value = x_span_value_list[span_x]
                width = x_span_list[span_x]
                x = current_x
                current_x += width
                if value > 0:
                    horizontal_image = image_rect_inside(horizontal_image, Rectangle(x, 0, width, 1), color_indicator)

            bottom_right_image = image_create(1, 1, color_background)

            if transformation_id == 'and':
                primary_image = image_and
            elif transformation_id == 'or':
                primary_image = image_or
            elif transformation_id == 'xor':
                primary_image = image_xor
            elif transformation_id == 'sum':
                primary_image = image_sum
            else:
                raise Exception(f"Unknown transformation_id: {transformation_id}")

            primary_image = image_replace_colors(primary_image, color_map_primary)

            if has_output_with_border:
                input_image_raw = grid2x2(primary_empty, vertical_image, horizontal_image, bottom_right_image)
                output_image_raw = grid2x2(primary_image, vertical_image, horizontal_image, bottom_right_image)
            else:
                input_image_raw = grid2x2(primary_empty, vertical_image, horizontal_image, bottom_right_image)
                output_image_raw = primary_image

            # Rotate
            rotate90_count = random.Random(iteration_seed + 30).randint(0, 3)
            input_image_raw = np.rot90(input_image_raw, rotate90_count)
            output_image_raw = np.rot90(output_image_raw, rotate90_count)

            # Palette
            input_image = image_replace_colors(input_image_raw, input_color_map)
            output_image = image_replace_colors(output_image_raw, output_color_map)

            break
        if input_image is None:
            raise Exception("Failed to create image")
        if output_image is None:
            raise Exception("Failed to create image")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image_randomized()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    j = seed % 11
    # j = (seed % 4) + 7
    if j == 0:
        task = generate_task_with_intersecting_spans(seed, 'empty_primary_area')
    elif j == 1:
        task = generate_task_with_intersecting_spans(seed, 'template_primary_area')
    elif j == 2:
        task = generate_task_with_intersecting_spans(seed, 'template_primary_area_without_border')
    elif j == 3:
        task = generate_task_with_intersecting_spans(seed, 'colored_primary_area_fill_template_border')
    elif j == 4:
        # This is identical to 'colored_primary_area_extract_horizontal', due to rotation of the tasks.
        task = generate_task_with_intersecting_spans(seed, 'colored_primary_area_extract_horizontal')
    elif j == 5:
        task = generate_task_with_template_lines(seed, 'output_with_border')
    elif j == 6:
        task = generate_task_with_template_lines(seed, 'output_without_border')
    elif j == 7:
        task = generate_task_with_alternate(seed, 'and')
    elif j == 8:
        task = generate_task_with_alternate(seed, 'or')
    elif j == 9:
        task = generate_task_with_alternate(seed, 'xor')
    elif j == 10:
        task = generate_task_with_alternate(seed, 'sum')
    transformation_id = task.metadata_task_id
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=2231000410,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
