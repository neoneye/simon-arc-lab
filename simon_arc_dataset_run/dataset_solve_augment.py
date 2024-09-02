# Augment existing ARC-AGI tasks
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
from simon_arc_lab.image_bresenham_line import *
from simon_arc_lab.image_mask import *
from simon_arc_lab.image_scale import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.benchmark import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_augment'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_augment.jsonl')

class BaseNode:
    def __init__(self):
        print("BaseNode.__init__")

    def apply_many(self, images: list[np.array]) -> list[np.array]:
        return [self.apply(image) for image in images]
    
    def apply(self, image: np.array) -> np.array:
        raise Exception("Not implemented")
    
    def name(self) -> str:
        raise Exception("Not implemented")

class NodeDoNothing(BaseNode):
    def __init__(self):
        print("NodeDoNothing.__init__")

    def apply(self, image: np.array) -> np.array:
        return image.copy()

    def name(self) -> str:
        return 'nop'

class NodeChain(BaseNode):
    def __init__(self, nodes: list[BaseNode]):
        self.nodes = nodes

    def apply_many(self, images: list[np.array]) -> list[np.array]:
        for node in self.nodes:
            images = node.apply_many(images)
        return images

    def apply(self, image: np.array) -> np.array:
        raise Exception("Not implemented for NodeChain")

    def name(self) -> str:
        names = [node.name() for node in self.nodes]
        return ','.join(names)

class NodeShuffleColors(BaseNode):
    def __init__(self, seed: int):
        print("NodeShuffleColors.__init__")
        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(seed + 3).shuffle(colors)
        color_map = {}
        for i, color in enumerate(colors):
            color_map[i] = color
        self.color_map = color_map

    def apply(self, image: np.array) -> np.array:
        return image_replace_colors(image, self.color_map)

    def name(self) -> str:
        return 'shuffle_colors'

class NodeRotateCW(BaseNode):
    def __init__(self):
        print("NodeRotateCW.__init__")

    def apply(self, image: np.array) -> np.array:
        return image_rotate_cw(image)

    def name(self) -> str:
        return 'rotate_cw'

class NodeScale(BaseNode):
    def __init__(self, x_up_down: str, x_scale: int, y_up_down: str, y_scale: int):
        print("NodeScale.__init__")
        self.x_up_down = x_up_down
        self.x_scale = x_scale
        self.y_up_down = y_up_down
        self.y_scale = y_scale

    def apply(self, image: np.array) -> np.array:
        input_image, output_image = image_scale(image, self.x_up_down, self.x_scale, self.y_up_down, self.y_scale)
        return output_image

    def name(self) -> str:
        return 'scale'

def generate_task_augmented(seed: int) -> Task:
    """
    Draw 2 crossing lines on an image.
    Highlight the lines that have a particular direction.
    Highlight the intersection of the 2 lines.
    """
    count_example = random.Random(seed + 1).randint(3, 4)
    count_test = random.Random(seed + 2).randint(1, 2)
    # count_test = 1
    task = Task()
    min_image_size = 3
    max_image_size = 10

    input_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(input_colors)
    color_map_input = {}
    for i, color in enumerate(input_colors):
        color_map_input[i] = color

    output_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 4).shuffle(output_colors)
    color_map_output = {}
    for i, color in enumerate(output_colors):
        color_map_output[i] = color

    available_line_ids = ['left_right', 'top_bottom', 'topleft_bottomright', 'topright_bottomleft']
    # pick two unique elements
    line_ids = random.Random(seed + 5).sample(available_line_ids, 2)

    # pick one of the line_ids as the output, or take the intersection
    available_output_ids = ['intersection'] + line_ids
    output_id = random.Random(seed + 6).choice(available_output_ids)

    available_input_ids = ['mask', 'color']
    input_id = random.Random(seed + 7).choice(available_input_ids)

    intersection_variant = random.Random(seed + 8).randint(0, 2)
    hide_intersection_point = intersection_variant == 1
    use_different_color_for_intersection_point = intersection_variant == 2

    pretty_line_ids = '_'.join(line_ids)
    pretty_line_ids = pretty_line_ids.replace('left_right', 'lr')
    pretty_line_ids = pretty_line_ids.replace('top_bottom', 'tb')
    pretty_line_ids = pretty_line_ids.replace('topleft_bottomright', 'tlbr')
    pretty_line_ids = pretty_line_ids.replace('topright_bottomleft', 'trbl')
    task.metadata_task_id = f'cross {pretty_line_ids} {input_id}{intersection_variant} {output_id}'

    has_diagonal_lines = 'topleft_bottomright' in line_ids or 'topright_bottomleft' in line_ids
    # print(f"has_diagonal_lines: {has_diagonal_lines}")

    for i in range(count_example+count_test):
        is_example = i < count_example
        is_test = i >= count_example
        input_image = None
        output_image = None
        for retry_index in range(100):
            iteration_seed = (retry_index * 10000) + (seed * 37) + (i * 9932342) + 101

            width = random.Random(iteration_seed + 1).randint(min_image_size, max_image_size)
            height = random.Random(iteration_seed + 2).randint(min_image_size, max_image_size)
            max_size = max(width, height)

            # This is where the 2 lines are going to intersect
            x = random.Random(iteration_seed + 7).randint(0, width-1)
            y = random.Random(iteration_seed + 8).randint(0, height-1)

            # It's not possible to determine where 2 lines intersect when it's on a corner. 
            # Skip the corners, when it's a test pair.
            # Allow corners for examples, so there is some ambiguity.
            is_x0 = x == 0
            is_x1 = x == width-1
            is_y0 = y == 0
            is_y1 = y == height-1
            is_x0y0 = is_x0 and is_y0
            is_x1y0 = is_x1 and is_y0
            is_x0y1 = is_x0 and is_y1
            is_x1y1 = is_x1 and is_y1
            position_in_corner = is_x0y0 or is_x1y0 or is_x0y1 or is_x1y1
            if is_test and has_diagonal_lines and position_in_corner:
                # print("Skip corner")
                continue

            accumulated_image_or = image_create(width, height, 0)
            accumulated_image_xor = image_create(width, height, 0)
            accumulated_image_intersection = image_create(width, height, 0)
            accumulated_image_lr = image_create(width, height, 0)
            accumulated_image_tb = image_create(width, height, 0)
            accumulated_image_tlbr = image_create(width, height, 0)
            accumulated_image_trbl = image_create(width, height, 0)

            drawing_image = image_create(width, height, 0)

            # Draw the 2 lines
            for j, line_id in enumerate(line_ids):
                image = image_create(width, height, 0)
                draw_color = j + 1

                if line_id == 'left_right':
                    image = image_bresenham_line(image, 0, y, width-1, y, 1)
                    drawing_image = image_bresenham_line(drawing_image, 0, y, width-1, y, draw_color)
                    accumulated_image_lr = image_mask_or(accumulated_image_lr, image)
                elif line_id == 'top_bottom':
                    image = image_bresenham_line(image, x, 0, x, height-1, 1)
                    drawing_image = image_bresenham_line(drawing_image, x, 0, x, height-1, draw_color)
                    accumulated_image_tb = image_mask_or(accumulated_image_tb, image)
                elif line_id == 'topleft_bottomright':
                    image = image_bresenham_line(image, x - max_size, y - max_size, x + max_size, y + max_size, 1)
                    drawing_image = image_bresenham_line(drawing_image, x - max_size, y - max_size, x + max_size, y + max_size, draw_color)
                    accumulated_image_tlbr = image_mask_or(accumulated_image_tlbr, image)
                elif line_id == 'topright_bottomleft':
                    image = image_bresenham_line(image, x + max_size, y - max_size, x - max_size, y + max_size, 1)
                    drawing_image = image_bresenham_line(drawing_image, x + max_size, y - max_size, x - max_size, y + max_size, draw_color)
                    accumulated_image_trbl = image_mask_or(accumulated_image_trbl, image)

                intersection_mask = image_mask_and(accumulated_image_or, image)
                accumulated_image_or = image_mask_or(accumulated_image_or, image)
                accumulated_image_xor = image_mask_xor(accumulated_image_xor, image)
                accumulated_image_intersection = image_mask_or(accumulated_image_intersection, intersection_mask)

            if hide_intersection_point:
                drawing_image[y, x] = 0
            elif use_different_color_for_intersection_point:
                drawing_image[y, x] = 3

            # Prepare input image
            input_image_raw = None
            if input_id == 'mask':
                input_image_raw = accumulated_image_xor
            elif input_id == 'color':
                input_image_raw = drawing_image
            input_image = image_replace_colors(input_image_raw, color_map_input)

            # We are not interested in an empty image
            histogram_input = Histogram.create_with_image(input_image)
            if histogram_input.number_of_unique_colors() < 2:
                continue

            # Prepare output image
            output_image_raw = None
            if output_id == 'left_right':
                output_image_raw = accumulated_image_lr
            elif output_id == 'top_bottom':
                output_image_raw = accumulated_image_tb
            elif output_id == 'topleft_bottomright':
                output_image_raw = accumulated_image_tlbr
            elif output_id == 'topright_bottomleft':
                output_image_raw = accumulated_image_trbl
            elif output_id == 'intersection':
                output_image_raw = accumulated_image_intersection
            output_image = image_replace_colors(output_image_raw, color_map_output)

            # We are not interested in an empty image
            histogram_output = Histogram.create_with_image(output_image)
            if histogram_output.number_of_unique_colors() < 2:
                continue

            break
        if (input_image is None) or (output_image is None):
            raise Exception("Failed to find a candidate images.")
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def generate_dataset_item_list(seed: int) -> list[dict]:
    transformation_id = 'cross'
    task = generate_task_augmented(seed)
    # task.show()
    dataset_items = generate_dataset_item_list_inner(seed, task, transformation_id)
    return dataset_items

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

def create_augmented_tasks(input_output: str, node_pre: BaseNode, node_transform: BaseNode, seed: int, task: Task) -> list[Task]:
    # Collect images for processing
    all_images = []
    if input_output == 'input':
        for i in range(task.count_examples):
            all_images.append(task.example_input(i))
        for i in range(task.count_tests):
            all_images.append(task.test_input(i))
    elif input_output == 'output':
        for i in range(task.count_examples):
            all_images.append(task.example_output(i))
    else:
        raise Exception(f"Unknown input_output: {input_output}")

    # Human readable name of the transformation
    name_pre = node_pre.name()
    name_transform = node_transform.name()

    # Split up many images into smaller chunks.
    random.Random(seed + 1).shuffle(all_images)
    groups = []
    count_images = len(all_images)
    if count_images <= 4:
        groups.append(all_images)
    elif count_images == 5:
        image0, image1, image2, image3, image4 = all_images
        group0 = [image0, image1, image2, image3]
        group1 = [image0, image1, image2, image4]
        groups.append(group0)
        groups.append(group1)
    elif count_images >= 6:
        image0, image1, image2, image3, image4, image5 = all_images[:6]
        group0 = [image0, image1, image2, image3]
        group1 = [image0, image1, image2, image4]
        group2 = [image0, image1, image2, image5]
        groups.append(group0)
        groups.append(group1)
        groups.append(group2)

    # Process groups of images
    augmented_tasks = []    
    for group_index, group_images in enumerate(groups):
        pair_count = len(group_images)

        # Preprocess images
        input_images = node_pre.apply_many(group_images)
        assert len(input_images) == pair_count

        # Transform images
        output_images = node_transform.apply_many(input_images)
        assert len(output_images) == pair_count

        # Create new task
        new_task = Task()
        new_task.metadata_task_id = f'{task.metadata_task_id} {input_output} group{group_index} pre_{name_pre} transform_{name_transform}'
        for pair_index in range(len(group_images)):
            input_image = input_images[pair_index]
            output_image = output_images[pair_index]
            new_task.append_pair(input_image, output_image, pair_index < pair_count - 1)
            
        new_task.shuffle_examples(seed + group_index)
        augmented_tasks.append(new_task)
    return augmented_tasks

augmented_tasks = []
for group_index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    taskset = TaskSet.load_directory(path_to_task_dir)

    # node_pre = NodeShuffleColors(123)
    node_pre = NodeDoNothing()
    node_rotate = NodeRotateCW()
    node_scale = NodeScale('up', 2, 'up', 2)
    node_transform = NodeChain([node_rotate, node_scale])

    for task_index, task in enumerate(taskset.tasks):
        iteration_seed = group_index * 1000000 + task_index * 1000
        new_tasks = create_augmented_tasks('input', node_pre, node_transform, iteration_seed + 1, task)
        augmented_tasks.extend(new_tasks)
        new_tasks = create_augmented_tasks('output', node_pre, node_transform, iteration_seed + 2, task)
        augmented_tasks.extend(new_tasks)

print(f"Number of augmented tasks: {len(augmented_tasks)}")

for task in augmented_tasks:
    task.show()

# generator = DatasetGenerator(
#     generate_dataset_item_list_fn=generate_dataset_item_list
# )
# generator.generate(
#     seed=1200023425,
#     max_num_samples=100000,
#     max_byte_size=1024*1024*100
# )
# # generator.inspect()
# generator.save(SAVE_FILE_PATH)
