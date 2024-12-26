from datetime import datetime
import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.task import Task
from simon_arc_lab.gallery_generator import gallery_generator_run

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

@dataclass
class PixelPosition:
    x: int
    y: int

@dataclass
class ImageSize:
    width: int
    height: int


class Node(ABC):
    @abstractmethod
    def compute_image(self, input_image: np.array) -> np.array:
        pass
    
class PixelNode(Node):
    @abstractmethod
    def compute_pixel(self, size: ImageSize, position: PixelPosition, pixel_value: int) -> int:
        return pixel_value
    
    def compute_image(self, input_image: np.array) -> np.array:
        height, width = input_image.shape
        image_size = ImageSize(width=width, height=height)
        output_image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                position = PixelPosition(x=x, y=y)
                output_image[y, x] = self.compute_pixel(image_size, position, input_image[y, x])
        return output_image

class DoNothingNode(Node):
    def compute_image(self, input_image: np.array) -> np.array:
        return input_image.copy()

class EdgePixelNode(PixelNode):
    def compute_pixel(self, size: ImageSize, position: PixelPosition, pixel_value: int) -> int:
        is_edge = position.x == 0 or position.y == 0 or position.x == size.width - 1 or position.y == size.height - 1
        if is_edge:
            return pixel_value
        return 0
    
class Solver:
    def __init__(self):
        self.nodes = [
            DoNothingNode(),
            # EdgePixelNode(),
        ]

    def process_training_pairs(self, task: Task, seed: int) -> int:
        if task.has_same_input_output_size_for_all_examples() == False:
            raise ValueError("Task has different input/output sizes for examples.")
        
        for train_index in range(task.count_examples):
            input = task.example_input(train_index)
            predicted_output = self.compute_output(input, seed)
            expected_output = task.example_output(train_index)
            if predicted_output.shape != expected_output.shape:
                return 0
            correct_pixels = np.sum(predicted_output == expected_output)
            return correct_pixels
    
    def process_test_pair(self, task: Task, test_index: int, seed: int) -> int:
        if task.has_same_input_output_size_for_all_examples() == False:
            raise ValueError("Task has different input/output sizes for examples.")
        
        input = task.test_input(test_index)
        predicted_output = self.compute_output(input, seed)
        expected_output = task.test_output(test_index)
        if predicted_output.shape != expected_output.shape:
            return 0
        correct_pixels = np.sum(predicted_output == expected_output)
        return correct_pixels
    
    def compute_output(self, input_image: np.array, seed: int) -> np.array:
        output_image = None
        for node_index, node in enumerate(self.nodes):
            if node_index == 0:
                buffer_input = input_image
            else:
                buffer_input = output_image
            output_image = node.compute_image(buffer_input)
        return output_image
    
number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    task_ids_to_keep = set()
    for task in taskset.tasks:
        if task.has_same_input_output_size_for_all_examples():
            task_ids_to_keep.add(task.metadata_task_id)
    taskset.keep_tasks_with_id(task_ids_to_keep)
    print(f"Number of tasks: {len(taskset.tasks)}")

    number_of_output_pixels_in_training_pairs = 0
    for task in taskset.tasks:
        for train_index in range(task.count_examples):
            output = task.example_output(train_index)
            height, width = output.shape
            number_of_output_pixels_in_training_pairs += width * height

    number_of_output_pixels_in_test_pairs = 0
    for task in taskset.tasks:
        for test_index in range(task.count_tests):
            output = task.test_output(test_index)
            height, width = output.shape
            number_of_output_pixels_in_test_pairs += width * height

    solver = Solver()

    number_of_iterations = 10
    for iteration_index in range(number_of_iterations):
        count_correct_pixels = 0
        for task in taskset.tasks:
            seed = iteration_index * 1000 + test_index
            count_correct_pixels += solver.process_training_pairs(task, seed)

        percent = count_correct_pixels * 100 / number_of_output_pixels_in_training_pairs
        percent_str = f"{percent:.2f}"
        print(f"Iteration {iteration_index+1} of {number_of_iterations}. total: {number_of_output_pixels_in_training_pairs} correct: {count_correct_pixels} ({percent_str}%)")

    test_count_correct_pixels = 0
    for task in taskset.tasks:
        for test_index in range(task.count_tests):
            seed = iteration_index * 1000 + test_index
            test_count_correct_pixels += solver.process_test_pair(task, test_index, seed)

    percent = test_count_correct_pixels * 100 / number_of_output_pixels_in_test_pairs
    percent_str = f"{percent:.2f}"
    print(f"Test total: {number_of_output_pixels_in_test_pairs} correct: {test_count_correct_pixels} ({percent_str}%)")