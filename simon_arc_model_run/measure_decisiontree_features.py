import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
import json
import numpy as np
import datetime
import time
import random
from random import sample
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_pixel_similarity import image_pixel_similarity_overall
from simon_arc_model.model import Model
from simon_arc_model.decision_tree_util import DecisionTreeUtil, DecisionTreeFeature

def featureset_id(features: set):
    names = sorted([feature.name for feature in features])
    return '_'.join(names)

class FeatureComboItem:
    def __init__(self, run_index: int, features: set):
        self.run_index = run_index
        self.features = features
    
    def feature_names_sorted(self):
        return sorted([feature.name for feature in self.features])

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

available_features = list(DecisionTreeFeature)
print(f"Number of features: {len(available_features)}")

available_feature_names = [feature.name for feature in available_features]
print(f"Feature names: {sorted(available_feature_names)}")

already_seen_featureids = set()
featurecomboitem_list = []
for i in range(20):
    features = None
    for retry_index in range(100):
        number_of_features_to_select = random.randint(1, 2)
        candidate_features = set(random.sample(available_features, number_of_features_to_select))
        fid = featureset_id(candidate_features)
        if fid in already_seen_featureids:
            continue
        already_seen_featureids.add(fid)
        features = candidate_features
        break
    if features is None:
        raise Exception("Failed to find a new feature set after 100 retries")
    featurecomboitem = FeatureComboItem(i + 1, features)
    featurecomboitem_list.append(featurecomboitem)

for featurecomboitem in featurecomboitem_list:
    print(f"FeatureComboItem {featurecomboitem.run_index}: {featurecomboitem.feature_names_sorted()}")


groupname_task_list = []
for (groupname, path_to_task_dir) in groupname_pathtotaskdir_list:
    taskset = TaskSet.load_directory(path_to_task_dir)

    pending_tasks = []
    number_of_tasks_with_different_input_output_size = 0
    for task in taskset.tasks:
        if DecisionTreeUtil.has_same_input_output_size_for_all_examples(task):
            pending_tasks.append(task)
        else:
            number_of_tasks_with_different_input_output_size += 1
    
    # truncate the list to 5 tasks
    pending_tasks = pending_tasks[:5]
    # print(f"Number of tasks with different input/output size: {number_of_tasks_with_different_input_output_size}")
    # print(f"Number of tasks with same input/output size: {len(pending_tasks)}")
    print(f"After filtering, number of tasks in group '{groupname}': {len(pending_tasks)}")
    groupname_task_list.append((groupname, pending_tasks))

# exit()
pbar_combo = tqdm(featurecomboitem_list, desc="Processing combos")
for combo in pbar_combo:
    save_dir = f'run_tasks_result/measure_decisiontree_features/{combo.run_index}'
    jsonl_filepath = f'{save_dir}/results.jsonl'

    number_of_items_in_list = len(groupname_pathtotaskdir_list)
    pbar_group = tqdm(groupname_task_list, desc="Processing groups")
    for (groupname, tasks) in pbar_group:
 
        correct_count = 0
        pbar_task = tqdm(pending_tasks, desc="Processing tasks")
        for task in pbar_task:
            pbar_task.set_description(f"{groupname} {task.metadata_task_id}")

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
                pbar_task.set_postfix({'correct': correct_count})

                count_good, count_total = image_pixel_similarity_overall(predicted_output, expected_output_image)
                count_bad = count_total - count_good
                if count_total > 0:
                    score = count_good * 100 // count_total
                else:
                    score = 0

                # if there was some problem, add it to the issues list
                jsonissues = []

                # An undesired issue is when the predicted output is the same as the input image
                is_same_as_input = np.array_equal(predicted_output, input_image)
                if is_correct == False and is_same_as_input:
                    # pbar.write("predicted output is the same as input")
                    jsonissues.append("predicted output is the same as input")

                # JSON representation of the prediction result
                jsondata = {
                    "correct": is_correct,
                    "score": score,
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
                    "issues": jsonissues,
                    "similarity": {
                        "score_type": "pixelwise",
                        "count_good": count_good,
                        "count_bad": count_bad,
                        "count_total": count_total,
                    },
                    "input": input_image.tolist(), # Convert numpy arrays to lists
                    "expected_output": expected_output_image.tolist(), # Convert numpy arrays to lists
                    "predicted_output": predicted_output.tolist(), # Convert numpy arrays to lists
                    "version": "simon_arc_lab measure_decisiontree_features 2024-oct-17"
                }

                # Save the result to a jsonl file
                append_to_jsonl_file(jsonl_filepath, jsondata)
