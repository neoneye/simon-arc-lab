from datetime import datetime
import sys
import os
import json
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.shape import *
from simon_arc_lab.pixel_connectivity import PixelConnectivity
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_string_representation import image_to_string

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data')),
    # ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
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

task_ids_of_interest = [
    # '0b17323b',
    '08573cc6',
    # '6c434453',
    # '21f83797',
    # '13713586',
    # '1c02dbbe',
    # '29700607',
    # '1a2e2828',
]

def analyze_image(image: np.array):
    background_color = 0
    connectivity = PixelConnectivity.NEAREST4
    ignore_color = background_color
    ignore_mask = (image == ignore_color).astype(np.uint8)
    connected_components = ConnectedComponent.find_objects_with_ignore_mask_inner(connectivity, image, ignore_mask)

    print(f"Number of connected components: {len(connected_components)}")

    for connected_component_item in connected_components:
        # print(f"Connected component item: {connected_component_item}")
        mask = connected_component_item.mask
        shape = image_find_shape(mask, verbose=False)
        if shape is None:
            print(f"Connected component item: {connected_component_item}")
            print(image_to_string(mask))
            # print(mask.tolist())
            continue
        print(f"Shape: {shape}")

def analyze_task(task: Task, index: int):
    for i in range(task.count_examples):
        # input_image = task.example_input(i)
        # analyze_image(input_image)
        output_image = task.example_output(i)
        analyze_image(output_image)


number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.keep_tasks_with_id(set(task_ids_of_interest), verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    pbar = tqdm(taskset.tasks, desc=f"Processing tasks in {groupname}", dynamic_ncols=True)
    for task in pbar:
        task_id = task.metadata_task_id
        pbar.set_postfix_str(f"Task: {task_id}")

        for test_index in range(task.count_tests):
            analyze_task(task, test_index)
            # filename = f'{task_id}_test{test_index}_prompt.md'
            # filepath = os.path.join(save_dir, filename)
            # with open(filepath, 'w') as f:
            #     f.write(prompt)
