import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
import json
import numpy as np
import datetime
import time
from simon_arc_lab.taskset import TaskSet
from simon_arc_model.model import Model
from simon_arc_model.decision_tree_util import DecisionTreeUtil, DecisionTreeFeature

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data/symmetry_rect_input_image_and_extract_a_particular_tile')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'stepwise')),
]

def append_to_jsonl_file(filepath, jsondata):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:  # 'a' for append mode
        json.dump(jsondata, f)
        f.write('\n')  # Ensure each entry is on a new line
        f.flush()  # Force writing to disk immediately


for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/measure_decisiontree_features/{groupname}'
    jsonl_filepath = f'{save_dir}/results.jsonl'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    pending_tasks = []
    number_of_tasks_with_different_input_output_size = 0
    for task in taskset.tasks:
        if DecisionTreeUtil.has_same_input_output_size_for_all_examples(task):
            pending_tasks.append(task)
        else:
            number_of_tasks_with_different_input_output_size += 1
    # print(f"Number of tasks with different input/output size: {number_of_tasks_with_different_input_output_size}")
    # print(f"Number of tasks with same input/output size: {len(pending_tasks)}")

    correct_count = 0
    pbar = tqdm(pending_tasks, desc="Processing tasks")
    for task in pbar:
        pbar.set_description(f"Task {task.metadata_task_id}")

        features = set()

        for test_index in range(task.count_tests):
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            start_time = time.perf_counter()

            predicted_output = DecisionTreeUtil.predict_output(
                task, 
                test_index, 
                previous_prediction=None,
                refinement_index=0, 
                noise_level=100,
                features=features,
            )

            end_time = time.perf_counter()
            elapsed_seconds = int(end_time - start_time)

            input_image = task.test_input(test_index)
            expected_output_image = task.test_output(test_index)

            is_correct = np.array_equal(predicted_output, expected_output_image)
            if is_correct:
                correct_count += 1
            pbar.set_postfix({'correct': correct_count})

            # JSON representation of the prediction result
            jsondata = {
                "correct": is_correct,
                "task_id": task.metadata_task_id,
                "path": task.metadata_path,
                "date": current_datetime,
                "elapsed_seconds": elapsed_seconds,
                "parameters": {
                    "CONNECTED_COMPONENTS_ALL8": True,
                    "HISTOGRAM_DIAGONAL": True,
                    "HISTOGRAM_ROWCOL": True,
                    "ROTATE45": True,
                    "EROSION_ALL8": False,
                    "LOOKAROUND_WINDOW_WIDTH": 3,
                    "LOOKAROUND_WINDOW_HEIGHT": 1
                },
                "test_index": test_index,
                "issues": [],  # Add any issues if relevant
                "similarity": ["AGREE_ON_COLOR 3", "SAME_UNIQUE_COLORS"],
                "input": input_image.tolist(), # Convert numpy arrays to lists
                "expected_output": expected_output_image.tolist(), # Convert numpy arrays to lists
                "predicted_output": predicted_output.tolist(), # Convert numpy arrays to lists
                "version": "simon_arc_lab feature_metric 2024-oct-17"
            }

            # Save the result to a jsonl file
            append_to_jsonl_file(jsonl_filepath, jsondata)
