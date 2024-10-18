import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
import json
from math import ceil
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

seed = 47

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
if False:
    features = set()
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(DecisionTreeFeature.COMPONENT_NEAREST4)
    features.add(DecisionTreeFeature.HISTOGRAM_DIAGONAL)
    features.add(DecisionTreeFeature.HISTOGRAM_ROWCOL)
    features.add(DecisionTreeFeature.HISTOGRAM_VALUE)
    features.add(DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(DecisionTreeFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if True:
    features = set()
    features.add(DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    features.add(DecisionTreeFeature.COMPONENT_NEAREST4)
    features.add(DecisionTreeFeature.HISTOGRAM_DIAGONAL)
    features.add(DecisionTreeFeature.HISTOGRAM_ROWCOL)
    features.add(DecisionTreeFeature.HISTOGRAM_VALUE)
    features.add(DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(DecisionTreeFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if True:
    features = set()
    features.add(DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    features.add(DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    features.add(DecisionTreeFeature.COMPONENT_NEAREST4)
    features.add(DecisionTreeFeature.HISTOGRAM_DIAGONAL)
    features.add(DecisionTreeFeature.HISTOGRAM_ROWCOL)
    features.add(DecisionTreeFeature.HISTOGRAM_VALUE)
    features.add(DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(DecisionTreeFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if True:
    features = set()
    features.add(DecisionTreeFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    features.add(DecisionTreeFeature.COMPONENT_NEAREST4)
    features.add(DecisionTreeFeature.HISTOGRAM_DIAGONAL)
    features.add(DecisionTreeFeature.HISTOGRAM_ROWCOL)
    features.add(DecisionTreeFeature.HISTOGRAM_VALUE)
    features.add(DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(DecisionTreeFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(3, features)
    featurecomboitem_list.append(featurecomboitem)

for i in range(60):
    run_index = i + 1
    for featurecomboitem in featurecomboitem_list:
        if run_index <= featurecomboitem.run_index:
            run_index = featurecomboitem.run_index + 1

    features = None
    for retry_index in range(100):
        iteration_seed = seed + (i + retry_index) * 10000
        number_of_features_to_select = random.Random(iteration_seed + 1).randint(2, 5)
        feature_list = random.Random(iteration_seed + 2).sample(available_features, number_of_features_to_select)
        candidate_features = set(feature_list)
        fid = featureset_id(candidate_features)
        if fid in already_seen_featureids:
            continue
        already_seen_featureids.add(fid)
        features = candidate_features
        break
    if features is None:
        raise Exception("Failed to find a new feature set after 100 retries")
    featurecomboitem = FeatureComboItem(run_index, features)
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
    
    random.Random(seed + 2).shuffle(pending_tasks)
    # truncate the list to a few tasks
    # pending_tasks = pending_tasks[:20]
    # print(f"Number of tasks with different input/output size: {number_of_tasks_with_different_input_output_size}")
    # print(f"Number of tasks with same input/output size: {len(pending_tasks)}")
    print(f"After filtering, number of tasks in group '{groupname}': {len(pending_tasks)}")
    groupname_task_list.append((groupname, pending_tasks))

for combo_index, combo in enumerate(featurecomboitem_list):
    print(f"Feature combo {combo_index+1} of {len(featurecomboitem_list)}, features: {combo.feature_names_sorted()}")
    save_dir = f'run_tasks_result/measure_decisiontree_features/{combo.run_index}'
    jsonl_filepath = f'{save_dir}/results.jsonl'
    summary_filepath = f'{save_dir}/summary.json'

    job_list = []
    for (groupname, tasks) in groupname_task_list:
        for task in tasks:
            job_list.append((groupname, task))

    feature_name_list = combo.feature_names_sorted()

    correct_count = 0
    total_elapsed_float = 0
    count_score9599 = 0
    count_score9094 = 0
    with tqdm(job_list, desc="Processing tasks", leave=False, position=0) as pbar:
        for (groupname, task) in pbar:
            desc = task.metadata_task_id
            # truncate string to 20 characters, if it's longer add ...
            desc = (desc[:20] + '...') if len(desc) > 20 else desc
            pbar.set_description(desc)

            for test_index in range(task.count_tests):
                current_datetime = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                
                start_time = time.perf_counter()

                predicted_output = DecisionTreeUtil.predict_output(
                    task, 
                    test_index, 
                    previous_prediction=None,
                    refinement_index=0, 
                    noise_level=100,
                    features=combo.features,
                )

                end_time = time.perf_counter()
                elapsed_float = end_time - start_time
                total_elapsed_float += elapsed_float
                elapsed_seconds = int(ceil(elapsed_float))

                input_image = task.test_input(test_index)
                expected_output_image = task.test_output(test_index)

                is_correct = np.array_equal(predicted_output, expected_output_image)
                if is_correct:
                    correct_count += 1
                pbar.set_postfix({'correct': correct_count})

                count_good, count_total = image_pixel_similarity_overall(predicted_output, expected_output_image)
                count_bad = count_total - count_good
                if count_total > 0:
                    score = count_good * 100 // count_total
                else:
                    score = 0

                if score == 100:
                    pass
                elif score >= 95:
                    count_score9599 += 1
                elif score >= 90:
                    count_score9094 += 1

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
                    "features": feature_name_list,
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
    # Print summary
    average_elapsed = total_elapsed_float / len(job_list)
    # format the average elapsed time as a string with 1 decimal
    total_elapsed_str = "{:.2f}".format(total_elapsed_float)
    average_elapsed_str = "{:.2f}".format(average_elapsed)
    print(f"correct: {correct_count} score95_99: {count_score9599} score90_94: {count_score9094} elapsed_total: {total_elapsed_str} elapsed_average: {average_elapsed_str}")
    summary_data = {
        "correct": correct_count,
        "score95_99": count_score9599,
        "score90_94": count_score9094,
        "elapsed_total": total_elapsed_str,
        "elapsed_average": average_elapsed_str,
        "task_count": len(job_list),
        "features": feature_name_list,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f)

print("Done")
