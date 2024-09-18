# Augment existing ARC-AGI tasks
#
# Work in progress. I have not yet found the secret sauce to augment the ARC-AGI tasks in a way that the model can learn from it.
# The model seems to gotten severely dumber after training with this.
#
# IDEA: what if I train the model for a shorter times, so there is less catastrophic forgetting. Would that help?
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
from simon_arc_lab.task_split import *
from simon_arc_lab.image_bresenham_line import *
from simon_arc_lab.image_mask import *
from simon_arc_lab.image_pad import *
from simon_arc_lab.image_scale import *
from simon_arc_lab.image_skew import *
from simon_arc_lab.histogram import *
from simon_arc_lab.benchmark import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_dataset.simon_solve_version1_names import SIMON_SOLVE_VERSION1_NAMES
from simon_arc_dataset.generate_solve import *
from simon_arc_dataset.dataset_generator import *

DATASET_NAMES = SIMON_SOLVE_VERSION1_NAMES
BENCHMARK_DATASET_NAME = 'solve_augment'
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_solve_augment.jsonl')

raise Exception("This is highly experimental code, and not ready for use. It's worsening the model severely at the moment. I may return to this code later.")

class ApplyManyError(ValueError):
    """Exception raised for errors in Node apply_many."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details

class BaseNode:
    def apply_many(self, images: list[np.array]) -> list[np.array]:
        return [self.apply(image) for image in images]
    
    def apply(self, image: np.array) -> np.array:
        raise Exception("Not implemented")
    
    def name(self) -> str:
        raise Exception("Not implemented")

class NodeDoNothing(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image.copy()

    def name(self) -> str:
        return 'nop'

class NodeChain(BaseNode):
    def __init__(self, node_list_with_optionals: list[Optional[BaseNode]]):
        # Remove the node's that are None
        self.nodes = [node for node in node_list_with_optionals if node is not None]

    def apply_many(self, images: list[np.array]) -> list[np.array]:
        for node in self.nodes:
            images = node.apply_many(images)
        return images

    def apply(self, image: np.array) -> np.array:
        raise Exception("Not implemented for NodeChain")

    def name(self) -> str:
        names = [node.name() for node in self.nodes]
        return ','.join(names)

class NodeRotateCW(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_rotate_cw(image)

    def name(self) -> str:
        return 'rotate_cw'

class NodeRotateCCW(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_rotate_ccw(image)

    def name(self) -> str:
        return 'rotate_ccw'

class NodeRotate180(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_rotate_ccw(image)

    def name(self) -> str:
        return 'rotate_180'

class NodeFlipX(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_flipx(image)

    def name(self) -> str:
        return 'flipx'

class NodeFlipY(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_flipy(image)

    def name(self) -> str:
        return 'flipy'

class NodeFlipA(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_flip_diagonal_a(image)

    def name(self) -> str:
        return 'flipa'

class NodeFlipB(BaseNode):
    def apply(self, image: np.array) -> np.array:
        return image_flip_diagonal_b(image)

    def name(self) -> str:
        return 'flipb'

# IDEA: Add noise to the scaled up image, so the model can learn to ignore the noise.
# IDEA: Sparse scale up, where each scaled up cell contains only 1 pixel from the original image. Least popular color.
# IDEA: Sparse scale up, where each scaled up cell contains only 1 pixel from the original image. Center pixel.
class NodeScaleUp(BaseNode):
    def __init__(self, x_scale: int, y_scale: int):
        self.x_scale = x_scale
        self.y_scale = y_scale

    def apply(self, image: np.array) -> np.array:
        input_image, output_image = image_scale(image, 'up', self.x_scale, 'up', self.y_scale)
        return output_image

    def name(self) -> str:
        return 'scaleup'

class NodePad(BaseNode):
    def __init__(self, seed: int, padding_color: int, min_pad_count: int, max_pad_count: int):
        self.seed = seed
        self.padding_color = padding_color
        self.min_pad_count = min_pad_count
        self.max_pad_count = max_pad_count

    def apply_many(self, images: list[np.array]) -> list[np.array]:
        new_images = []
        for image_index, image in enumerate(images):
            new_image = image_pad_random(
                image, 
                self.seed + image_index * 1000, 
                self.padding_color, 
                self.min_pad_count, 
                self.max_pad_count
            )
            new_images.append(new_image)
        return new_images

    def name(self) -> str:
        return 'pad'

class NodeSwapColors(BaseNode):
    def apply_many(self, images: list[np.array]) -> list[np.array]:
        histogram_union = Histogram.empty()
        for image in images:
            histogram = Histogram.create_with_image(image)
            if histogram.number_of_unique_colors() != 2:
                raise ApplyManyError("Not all images are two color images")
            histogram_union = histogram_union.add(histogram)
        # print(f"Union of histograms: {histogram_union.pretty()}")

        if histogram_union.number_of_unique_colors() != 2:
            raise ApplyManyError("Not all images have the same two colors")

        color_count_list = histogram_union.sorted_color_count_list()
        color0 = color_count_list[0][0]
        color1 = color_count_list[1][0]

        color_map = {
            color0: color1,
            color1: color0,
        }
        # print("swapping colors")
        result_images = []
        for image in images:
            new_image = image_replace_colors(image, color_map)
            result_images.append(new_image)

        return result_images

    def name(self) -> str:
        return 'swap_colors'

class NodeSkew(BaseNode):
    def __init__(self, padding_color: int, direction: SkewDirection) -> None:
        super().__init__()
        self.padding_color = padding_color
        self.direction = direction

    def apply(self, image: np.array) -> np.array:
        return image_skew(image, self.padding_color, self.direction)

    def name(self) -> str:
        return 'skew'

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('diva', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-diva/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

def create_task_from_images(images: list[np.array], node_input: BaseNode, node_output: BaseNode, task_id_prefix: str, seed: int) -> Task:
    pair_count = len(images)
    if pair_count < 3:
        raise ValueError("Need at least 3 images to create a task")

    # Post process input images, such as scaling up, padding, adding noise
    input_images = node_input.apply_many(images)
    assert len(input_images) == pair_count

    # Transform images
    output_images = node_output.apply_many(images)
    assert len(output_images) == pair_count

    # Tasks where the input/output images are the same are not interesting.
    for pair_index in range(pair_count):
        input_image = input_images[pair_index]
        output_image = output_images[pair_index]
        if np.array_equal(input_image, output_image):
            raise ValueError("one or more Input/Output images are the same")

    # Human readable name of the transformation
    name_input = node_input.name()
    name_transform = node_output.name()

    # Shuffle colors
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed).shuffle(colors)
    color_map = {}
    for i, color in enumerate(colors):
        color_map[i] = color

    # Create new task
    new_task = Task()
    new_task.metadata_task_id = f'{task_id_prefix} input_{name_input} trans_{name_transform}'
    for pair_index in range(pair_count):
        input_image_raw = input_images[pair_index]
        output_image_raw = output_images[pair_index]
        input_image = image_replace_colors(input_image_raw, color_map)
        output_image = image_replace_colors(output_image_raw, color_map)
        new_task.append_pair(input_image, output_image, pair_index < pair_count - 1)

    return new_task

def create_task_from_taskimages(input_output: str, seed: int, task: Task) -> Optional[Task]:
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

    # Split up many images into smaller chunks.
    random.Random(seed + 1).shuffle(all_images)

    # Truncate to N images
    max_pair_count = 4
    truncated_images = all_images[:max_pair_count]

    histogram = Histogram.create_with_image_list(truncated_images)
    available_colors = histogram.available_colors()
    random.Random(seed + 2).shuffle(available_colors)

    # Pick colors for padding that doesn't clash with the colors already used by the images.
    color_pad_input = None
    color_pad_output = None
    if len(available_colors) >= 2:
        color0 = available_colors[0]
        color1 = available_colors[1]
        if random.Random(seed + 2).randint(0, 1) == 0:
            # Use 2 different colors
            color_pad_input = color0
            color_pad_output = color1
        else:
            # Use the same color
            color_pad_input = color0
            color_pad_output = color0
    elif len(available_colors) == 1:
        # Use the same color
        color0 = available_colors[0]
        color_pad_input = color0
        color_pad_output = color0

    node_input = create_node_input(seed * 910177 + 5, truncated_images, color_pad_input)
    node_output = create_node_output(seed * 130131 + 1, truncated_images, color_pad_output)
    # print(f"node: {node_input.name()} {node_output.name()}")

    task_id_prefix = f'{task.metadata_task_id} {input_output}'
    try:
        new_task = create_task_from_images(truncated_images, node_input, node_output, task_id_prefix, seed + 5)
    except ApplyManyError as e:
        print(f"create_task_from_images. Error: {e}")
        return None
    except ValueError as e:
        print(f"create_task_from_images. ValueError: {e}")
        return None
    new_task.shuffle_examples(seed + 6)
    return new_task

def create_multiple_tasks_from_taskimages(input_output: str, seed: int, task: Task, number_of_permutation: int) -> list[Task]:
    task_list = []
    for i in range(number_of_permutation):
        for retry_index in range(100):
            new_task = create_task_from_taskimages(input_output, seed + i * 10383838 + retry_index * 38919923, task)
            if new_task is None:
                continue
            task_list.append(new_task)
            break
    return task_list

def create_augmented_task(task: Task, node_input: BaseNode, node_output: BaseNode, seed: int) -> Task:
    """
    Create a new task by applying input and output nodes to the original task.
    """
    n = task.count_examples + task.count_tests

    input_images = []
    output_images = []
    for i in range(n):
        input_images.append(task.input_images[i].copy())
        output_images.append(task.output_images[i].copy())

    try:
        new_input_images = node_input.apply_many(input_images)
        new_output_images = node_output.apply_many(output_images)
    except ApplyManyError as e:
        print(f"create_augmented_task. Error: {e}")
        return None

    assert len(new_input_images) == n
    assert len(new_output_images) == n

    # Shuffle colors
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed).shuffle(colors)
    color_map = {}
    for i, color in enumerate(colors):
        color_map[i] = color

    new_task = Task()
    new_task.metadata_task_id = f'{task.metadata_task_id} {node_input.name()} {node_output.name()}'

    for i in range(n):
        is_example = i < task.count_examples
        input_image = image_replace_colors(new_input_images[i], color_map)
        output_image = image_replace_colors(new_output_images[i], color_map)
        new_task.append_pair(input_image, output_image, is_example)

    return new_task

def taskspecific_create_node_input(seed: int) -> BaseNode:
    j = random.Random(seed + 1).randint(0, 3)
    node_scaleup = None
    if j == 0:
        node_scaleup = NodeScaleUp(2, 2)
    elif j == 1:
        node_scaleup = NodeScaleUp(3, 3)
    elif j == 2:
        node_scaleup = NodeScaleUp(4, 4)

    node_list = [node_scaleup]

    return NodeChain(node_list)

def taskspecific_create_node_output(seed: int) -> BaseNode:
    j = random.Random(seed + 1).randint(0, 3)
    node_scaleup = None
    if j == 0:
        node_scaleup = NodeScaleUp(2, 2)
    elif j == 1:
        node_scaleup = NodeScaleUp(3, 3)
    elif j == 2:
        node_scaleup = NodeScaleUp(4, 4)

    node_list = [node_scaleup]

    return NodeChain(node_list)

def create_multiple_augmented_tasks_from_task(seed: int, task: Task, number_of_permutations: int) -> list[Task]:
    max_example_pairs = 3
    splitted_tasks = task_split(task, seed + 3, max_example_pairs, number_of_permutations)

    task_list = []
    for task_index, task in enumerate(splitted_tasks):
        if len(task_list) >= number_of_permutations:
            break
        for i in range(number_of_permutations):
            iteration_seed = seed + i * 10383838 + task_index * 38919923

            node_input = taskspecific_create_node_input(iteration_seed + 1)
            node_output = taskspecific_create_node_output(iteration_seed + 2)

            new_task = create_augmented_task(task, node_input, node_output, iteration_seed + 3)
            if new_task is None:
                continue
            new_task.shuffle_examples(iteration_seed + 4)
            task_list.append(new_task)
            break
    return task_list

def create_node_input(seed: int, images: list[np.array], pad_color: Optional[int]) -> BaseNode:
    """
    :param seed: The seed for the random number generator
    :param images: The images to be transformed
    :param pad_color: The color to use for padding, a color that doesn't clash with the colors already used by the images.
    :return: A node that is configured for transforming images
    """
    j = random.Random(seed + 1).randint(0, 7)
    if j == 0:
        node_rotateflip = NodeRotateCW()
    elif j == 1:
        node_rotateflip = NodeRotateCCW()
    elif j == 2:
        node_rotateflip = NodeRotate180()
    elif j == 3:
        node_rotateflip = NodeFlipX()
    elif j == 4:
        node_rotateflip = NodeFlipY()
    elif j == 5:
        node_rotateflip = NodeFlipA()
    elif j == 6:
        node_rotateflip = NodeFlipB()
    else:
        node_rotateflip = None

    j = random.Random(seed + 2).randint(0, 8)
    if j == 0:
        node_scale = NodeScaleUp(2, 2)
    elif j == 1:
        node_scale = NodeScaleUp(3, 3)
    elif j == 2:
        node_scale = NodeScaleUp(1, 2)
    elif j == 3:
        node_scale = NodeScaleUp(2, 1)
    elif j == 4:
        node_scale = NodeScaleUp(1, 3)
    elif j == 5:
        node_scale = NodeScaleUp(3, 1)
    elif j == 6:
        node_scale = NodeScaleUp(2, 3)
    elif j == 7:
        node_scale = NodeScaleUp(3, 2)
    else:
        node_scale = None

    node_skew_or_pad = None
    if pad_color is not None:
        j = random.Random(seed + 4).randint(0, 5)
        if j == 0:
            node_skew_or_pad = NodeSkew(pad_color, SkewDirection.UP)
        elif j == 1:
            node_skew_or_pad = NodeSkew(pad_color, SkewDirection.DOWN)
        elif j == 2:
            node_skew_or_pad = NodeSkew(pad_color, SkewDirection.LEFT)
        elif j == 3:
            node_skew_or_pad = NodeSkew(pad_color, SkewDirection.RIGHT)
        elif j == 4:
            node_skew_or_pad = NodePad(seed, pad_color, 1, 5)

    node_list_with_optionals = [node_rotateflip, node_scale, node_skew_or_pad]
    node_transform = NodeChain(node_list_with_optionals)
    return node_transform

def create_node_output(seed: int, images: list[np.array], pad_color: Optional[int]) -> BaseNode:
    """
    :param seed: The seed for the random number generator
    :param images: The images to be transformed
    :param pad_color: The color to use for padding, a color that doesn't clash with the colors already used by the images.
    :return: A node that is configured for transforming images
    """
    j = random.Random(seed + 1).randint(0, 7)
    if j == 0:
        node_rotateflip = NodeRotateCW()
    elif j == 1:
        node_rotateflip = NodeRotateCCW()
    elif j == 2:
        node_rotateflip = NodeRotate180()
    elif j == 3:
        node_rotateflip = NodeFlipX()
    elif j == 4:
        node_rotateflip = NodeFlipY()
    elif j == 5:
        node_rotateflip = NodeFlipA()
    elif j == 6:
        node_rotateflip = NodeFlipB()
    else:
        node_rotateflip = None

    j = random.Random(seed + 2).randint(0, 8)
    if j == 0:
        node_scale = NodeScaleUp(2, 2)
    elif j == 1:
        node_scale = NodeScaleUp(3, 3)
    elif j == 2:
        node_scale = NodeScaleUp(1, 2)
    elif j == 3:
        node_scale = NodeScaleUp(2, 1)
    elif j == 4:
        node_scale = NodeScaleUp(1, 3)
    elif j == 5:
        node_scale = NodeScaleUp(3, 1)
    elif j == 6:
        node_scale = NodeScaleUp(2, 3)
    elif j == 7:
        node_scale = NodeScaleUp(3, 2)
    else:
        node_scale = None

    node_skew = None
    if pad_color is not None:
        j = random.Random(seed + 6).randint(0, 4)
        if j == 0:
            node_skew = NodeSkew(pad_color, SkewDirection.UP)
        elif j == 1:
            node_skew = NodeSkew(pad_color, SkewDirection.DOWN)
        elif j == 2:
            node_skew = NodeSkew(pad_color, SkewDirection.LEFT)
        elif j == 3:
            node_skew = NodeSkew(pad_color, SkewDirection.RIGHT)
    
    node_list_with_optionals = [node_rotateflip, node_scale, node_skew]
    node_transform = NodeChain(node_list_with_optionals)
    return node_transform

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def mutated_tasks_from_task(task: Task, seed: int) -> list[Task]:

    number_of_permutations = 3

    new_tasks_input = create_multiple_tasks_from_taskimages('input', seed + 1, task, number_of_permutations)
    new_tasks_output = create_multiple_tasks_from_taskimages('output', seed + 2, task, number_of_permutations)
    augmented_tasks = create_multiple_augmented_tasks_from_task(seed + 3, task, number_of_permutations)

    mutated_tasks = new_tasks_input + new_tasks_output + augmented_tasks

    print(f"Number of input tasks: {len(new_tasks_input)}")
    print(f"Number of output tasks: {len(new_tasks_output)}")
    print(f"Number of augmented tasks: {len(augmented_tasks)}")
    print(f"Number of mutations: {len(mutated_tasks)} from task {task.metadata_task_id}")
    return mutated_tasks

original_tasks = []
for group_index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    taskset = TaskSet.load_directory(path_to_task_dir)
    for task in taskset.tasks:
        original_tasks.append(task)

count_original_tasks = len(original_tasks)
print(f"Number of original tasks: {count_original_tasks}")

# for task in original_tasks:
#     task.show()

def generate_dataset_item_list(seed: int) -> list[dict]:
    accumulated_tasks = []
    for task_index, task in enumerate(original_tasks):
        new_tasks = mutated_tasks_from_task(task, seed + task_index * 10010101)
        accumulated_tasks.extend(new_tasks)

    print(f"Number of tasks: {len(accumulated_tasks)}")

    accumulated_dataset_items = []
    for task_index, task in enumerate(accumulated_tasks):
        if task.total_pixel_count() > 1000:
            continue
        transformation_id = task.metadata_task_id
        task.show()
        dataset_items = generate_dataset_item_list_inner(seed + task_index * 100053523, task, transformation_id)
        accumulated_dataset_items.extend(dataset_items)
    return accumulated_dataset_items

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1200023425,
    max_num_samples=1000,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
