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
    def apply(self, image: np.array) -> np.array:
        # IDEA: if the output is identical to the input, then raise an exception.
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

class NodeScale(BaseNode):
    def __init__(self, x_up_down: str, x_scale: int, y_up_down: str, y_scale: int):
        self.x_up_down = x_up_down
        self.x_scale = x_scale
        self.y_up_down = y_up_down
        self.y_scale = y_scale

    def apply(self, image: np.array) -> np.array:
        input_image, output_image = image_scale(image, self.x_up_down, self.x_scale, self.y_up_down, self.y_scale)
        return output_image

    def name(self) -> str:
        return 'scale'

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
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

def create_task_from_images(images: list[np.array], node_pre: BaseNode, node_transform: BaseNode, node_post_input: BaseNode, task_id_prefix: str) -> Task:
    pair_count = len(images)

    # Prepare input images
    prepared_input_images = node_pre.apply_many(images)
    assert len(prepared_input_images) == pair_count

    # Transform images
    output_images = node_transform.apply_many(prepared_input_images)
    assert len(output_images) == pair_count

    # Post process input images, such as scaling up, padding, adding noise
    input_images = node_post_input.apply_many(prepared_input_images)
    assert len(input_images) == pair_count

    # Tasks where the input/output images are the same are not interesting.
    for pair_index in range(pair_count):
        input_image = input_images[pair_index]
        output_image = output_images[pair_index]
        if np.array_equal(input_image, output_image):
            raise ValueError("one or more Input/Output images are the same")

    # Human readable name of the transformation
    name_pre = node_pre.name()
    name_transform = node_transform.name()
    name_post_input = node_post_input.name()

    # Create new task
    new_task = Task()
    new_task.metadata_task_id = f'{task_id_prefix} pre_{name_pre} post_{name_post_input} trans_{name_transform}'
    for pair_index in range(pair_count):
        input_image = input_images[pair_index]
        output_image = output_images[pair_index]
        new_task.append_pair(input_image, output_image, pair_index < pair_count - 1)

    return new_task

def create_multiple_tasks_from_taskimages(input_output: str, node_pre: BaseNode, node_transform: BaseNode, node_post_input: BaseNode, seed: int, task: Task) -> list[Task]:
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
        task_id_prefix = f'{task.metadata_task_id} {input_output}'
        try:
            new_task = create_task_from_images(group_images, node_pre, node_transform, node_post_input, task_id_prefix)
        except ApplyManyError as e:
            print(f"create_task_from_images. Error: {e}")
            continue
        except ValueError as e:
            print(f"create_task_from_images. ValueError: {e}")
            continue
        new_task.shuffle_examples(seed + group_index)
        augmented_tasks.append(new_task)
    return augmented_tasks

def create_augmented_task(task: Task, node_input: BaseNode, node_output: BaseNode) -> Task:
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

    new_task = Task()
    new_task.metadata_task_id = f'{task.metadata_task_id} {node_input.name()} {node_output.name()}'

    for i in range(n):
        is_example = i < task.count_examples
        new_task.append_pair(new_input_images[i], new_output_images[i], is_example)

    return new_task

NUMBER_OF_PERMUTATIONS_PRE = 2
# NUMBER_OF_PERMUTATIONS_TRANSFORM = 2 * 9 * 4 * 5 * 5
NUMBER_OF_PERMUTATIONS_TRANSFORM = 5
NUMBER_OF_PERMUTATIONS_INPUT_POST = 9
NUMBER_OF_PERMUTATIONS_INPUT_OUTPUT = 2
NUMBER_OF_PERMUTATIONS_TOTAL = NUMBER_OF_PERMUTATIONS_PRE * NUMBER_OF_PERMUTATIONS_TRANSFORM * NUMBER_OF_PERMUTATIONS_INPUT_POST * NUMBER_OF_PERMUTATIONS_INPUT_OUTPUT

def permuted_node_pre(permutation: int) -> BaseNode:
    j = permutation % 2
    permutation = permutation // 2
    if j == 0:
        node_shuffle_colors = NodeShuffleColors(permutation % 100)
    else:
        node_shuffle_colors = None

    node_list_with_optionals = [node_shuffle_colors]
    # Remove the node's that are None
    node_list = [node for node in node_list_with_optionals if node is not None]

    node_transform = NodeChain(node_list)
    return node_transform

def permuted_node_transform(permutation: int) -> BaseNode:
    j = permutation % 2
    permutation = permutation // 2
    j = 1
    if j == 0:
        node_swap_colors = NodeSwapColors()
    else:
        node_swap_colors = None

    j = permutation % 9
    permutation = permutation // 8
    j = 8
    if j == 0:
        node_scale = NodeScale('up', 2, 'up', 2)
    elif j == 1:
        node_scale = NodeScale('up', 3, 'up', 3)
    elif j == 2:
        node_scale = NodeScale('up', 1, 'up', 2)
    elif j == 3:
        node_scale = NodeScale('up', 2, 'up', 1)
    elif j == 4:
        node_scale = NodeScale('up', 1, 'up', 3)
    elif j == 5:
        node_scale = NodeScale('up', 3, 'up', 1)
    elif j == 6:
        node_scale = NodeScale('up', 2, 'up', 3)
    elif j == 7:
        node_scale = NodeScale('up', 3, 'up', 2)
    else:
        node_scale = None

    j = permutation % 4
    j = 3
    permutation = permutation // 4
    if j == 0:
        node_rotate = NodeRotateCW()
    elif j == 1:
        node_rotate = NodeRotateCCW()
    elif j == 2:
        node_rotate = NodeRotate180()
    elif j == 3:
        node_rotate = None

    j = permutation % 5
    j = 4
    permutation = permutation // 5
    if j == 0:
        node_flip = NodeFlipX()
    elif j == 1:
        node_flip = NodeFlipY()
    elif j == 2:
        node_flip = NodeFlipA()
    elif j == 3:
        node_flip = NodeFlipB()
    else:
        node_flip = None

    j = permutation % 5
    permutation = permutation // 5
    # IDEA: Pick a skew_color that doesn't clash too much with the payload of the image.
    skew_color = permutation % 10
    permutation = permutation // 10
    if j == 0:
        node_skew = NodeSkew(skew_color, SkewDirection.UP)
    elif j == 1:
        node_skew = NodeSkew(skew_color, SkewDirection.DOWN)
    elif j == 2:
        node_skew = NodeSkew(skew_color, SkewDirection.LEFT)
    elif j == 3:
        node_skew = NodeSkew(skew_color, SkewDirection.RIGHT)
    else:
        node_skew = None
    
    node_list_with_optionals = [node_swap_colors, node_rotate, node_scale, node_flip, node_skew]
    # Remove the node's that are None
    node_list = [node for node in node_list_with_optionals if node is not None]

    node_transform = NodeChain(node_list)
    return node_transform

def permuted_node_input_post(permutation: int) -> BaseNode:
    j = permutation % 9
    j = 8
    permutation = permutation // 9
    if j == 0:
        node_scale = NodeScale('up', 2, 'up', 2)
    elif j == 1:
        node_scale = NodeScale('up', 3, 'up', 3)
    elif j == 2:
        node_scale = NodeScale('up', 1, 'up', 2)
    elif j == 3:
        node_scale = NodeScale('up', 2, 'up', 1)
    elif j == 4:
        node_scale = NodeScale('up', 1, 'up', 3)
    elif j == 5:
        node_scale = NodeScale('up', 3, 'up', 1)
    elif j == 6:
        node_scale = NodeScale('up', 2, 'up', 3)
    elif j == 7:
        node_scale = NodeScale('up', 3, 'up', 2)
    else:
        node_scale = None

    node_list_with_optionals = [node_scale]
    # Remove the node's that are None
    node_list = [node for node in node_list_with_optionals if node is not None]

    node_transform = NodeChain(node_list)
    return node_transform

original_tasks = []
for group_index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    taskset = TaskSet.load_directory(path_to_task_dir)
    for task in taskset.tasks:
        original_tasks.append(task)

count_original_tasks = len(original_tasks)
print(f"Number of original tasks: {count_original_tasks}")

# for task in original_tasks:
#     task.show()

def generate_dataset_item_list_inner(seed: int, task: Task, transformation_id: str) -> list[dict]:
    builder = DatasetItemListBuilder(seed, task, DATASET_NAMES, BENCHMARK_DATASET_NAME, transformation_id)
    builder.append_image()
    return builder.dataset_items()

def mutated_tasks_from_task(task: Task, seed: int) -> list[Task]:
    permutation = seed * 103483827531

    node_pre = permuted_node_pre(permutation)
    permutation = permutation // NUMBER_OF_PERMUTATIONS_PRE

    node_transform = permuted_node_transform(permutation)
    permutation = permutation // NUMBER_OF_PERMUTATIONS_TRANSFORM

    node_input_post = permuted_node_input_post(permutation)
    permutation = permutation // NUMBER_OF_PERMUTATIONS_INPUT_POST

    # print(f"node: {node_pre.name()} {node_transform.name()} {node_input_post.name()}")

    new_tasks_input = create_multiple_tasks_from_taskimages('input', node_pre, node_transform, node_input_post, permutation, task)
    # IDEA: This currently creates just 1 task. Create more than 1 task, N permutations.
    permutation = permutation // 2

    new_tasks_output = create_multiple_tasks_from_taskimages('output', node_pre, node_transform, node_input_post, permutation, task)
    # IDEA: This currently creates just 1 task. Create more than 1 task, N permutations.
    permutation = permutation // 2

    splitted_tasks = task_split(task, permutation, 3, 3)

    node_input = NodeScale('up', 2, 'up', 2)
    node_output = NodeShuffleColors(42)
    augmented_tasks = []
    for task in splitted_tasks:
        augmented_task = create_augmented_task(task, node_input, node_output)
        if augmented_task is not None:
            augmented_tasks.append(augmented_task)

    mutated_tasks = new_tasks_input + new_tasks_output + augmented_tasks

    print(f"Number of input tasks: {len(new_tasks_input)}")
    print(f"Number of output tasks: {len(new_tasks_output)}")
    print(f"Number of augmented tasks: {len(augmented_tasks)}")
    print(f"Number of mutations: {len(mutated_tasks)} from task {task.metadata_task_id}")
    return mutated_tasks

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

max_num_samples = min(100000, count_original_tasks * NUMBER_OF_PERMUTATIONS_TOTAL)
print(f"count_original_tasks: {count_original_tasks}")
print(f"NUMBER_OF_PERMUTATIONS_TRANSFORM: {NUMBER_OF_PERMUTATIONS_TRANSFORM}")
print(f"NUMBER_OF_PERMUTATIONS_INPUT_POST: {NUMBER_OF_PERMUTATIONS_INPUT_POST}")
print(f"NUMBER_OF_PERMUTATIONS_TOTAL: {NUMBER_OF_PERMUTATIONS_TOTAL}")
print(f"max_num_samples: {max_num_samples}")

generator = DatasetGenerator(
    generate_dataset_item_list_fn=generate_dataset_item_list
)
generator.generate(
    seed=1200023425,
    max_num_samples=max_num_samples,
    max_byte_size=1024*1024*100
)
# generator.inspect()
generator.save(SAVE_FILE_PATH)
