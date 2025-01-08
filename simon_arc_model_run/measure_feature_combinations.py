"""
Measure the performance of a decision tree model with different feature sets.
It's not possible to test all feature combos, so a few random combinations are selected.
It takes around 30m-2h to process all 800 puzzles in the ARC-AGI dataset.
Puzzles that have previously never been solved are especially interesting.
Puzzles that have been solved with a score of 100 are not interesting to solve again.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
import json
from math import ceil
import numpy as np
from datetime import datetime
import time
import random
from random import sample
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_pixel_similarity import image_pixel_similarity_overall
from simon_arc_model.image_feature import ImageFeature
from simon_arc_model.track_incorrect_prediction import TrackIncorrectPrediction
from simon_arc_model.model_beta1 import ModelBeta1

def featureset_id(features: set):
    return ImageFeature.names_sorted_and_joined(features, separator='_')

class FeatureComboItem:
    def __init__(self, run_index: int, features: set):
        self.run_index = run_index
        self.features = features
    
    def feature_names_sorted(self):
        return sorted([feature.name for feature in self.features])

seed = 55

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

incorrect_predictions_jsonl_path = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
# incorrect_predictions_jsonl_path = None

if incorrect_predictions_jsonl_path is not None:
    track_incorrect_prediction = TrackIncorrectPrediction.load_from_jsonl(incorrect_predictions_jsonl_path)
else:
    track_incorrect_prediction = None


def append_to_jsonl_file(filepath, jsondata):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:  # 'a' for append mode
        json.dump(jsondata, f)
        f.write('\n')  # Ensure each entry is on a new line
        f.flush()  # Force writing to disk immediately


for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

print(f"Number of all features: {len(list(ImageFeature))}")

available_features = list(ModelBeta1.supported_features())
print(f"Number of features supported by this model: {len(available_features)}")

available_feature_names_str = ImageFeature.names_sorted_and_joined(available_features, separator=', ')
print(f"Feature names: {available_feature_names_str}")

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
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.HISTOGRAM_DIAGONAL)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(ImageFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.HISTOGRAM_DIAGONAL)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(ImageFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.HISTOGRAM_DIAGONAL)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(ImageFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.HISTOGRAM_DIAGONAL)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(ImageFeature.BOUNDING_BOXES)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(3, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.CORNER)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL)
    features.add(ImageFeature.ROTATE45)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.EROSION_ALL8)
    features.add(ImageFeature.HISTOGRAM_DIAGONAL)
    features.add(ImageFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4)
    features.add(ImageFeature.ROTATE45)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.BOUNDING_BOXES)
    features.add(ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR)
    features.add(ImageFeature.EROSION_DIAGONAL)
    features.add(ImageFeature.HISTOGRAM_VALUE)
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.BOUNDING_BOXES)
    features.add(ImageFeature.COMPONENT_NEAREST4)
    features.add(ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR)
    features.add(ImageFeature.NUMBER_OF_UNIQUE_COLORS_ALL9)
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND)
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(1, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.EROSION_ROWCOL)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.NUMBER_OF_UNIQUE_COLORS_ALL9)
    features.add(ImageFeature.SUPPRESS_CENTER_PIXEL_ONCE)
    fid = featureset_id(features)
    already_seen_featureids.add(fid)
    featurecomboitem = FeatureComboItem(2, features)
    featurecomboitem_list.append(featurecomboitem)

if False:
    features = set()
    features.add(ImageFeature.CORNER)
    features.add(ImageFeature.HISTOGRAM_ROWCOL)
    features.add(ImageFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4)
    features.add(ImageFeature.ROTATE45)
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
        number_of_features_to_select = 3
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

# The goal is to solve puzzles that have never been solved before.
# Tasks that have previously been solved fine, are not interesting to solve again.
# the csv is a list of task ids that have a score of 100, there are no other columns in the file, so it can be split by \n
csv_file = 'finetune/taskids_with_score100.csv'
taskids_to_ignore = set()
if os.path.isfile(csv_file):
    with open(csv_file, 'r') as f:
        taskids = f.read().split('\n')
        taskids_to_ignore = set(taskids)
print(f"Number of task ids to ignore: {len(taskids_to_ignore)}")

datasetid_groupname_task_list = []
for (dataset_id, groupname, path_to_task_dir) in datasetid_groupname_pathtotaskdir_list:
    taskset_all = TaskSet.load_directory(path_to_task_dir)
    new_tasks = []
    for task in taskset_all.tasks:
        if task.metadata_task_id in taskids_to_ignore:
            # print(f"Ignoring task id: {task.metadata_task_id}")
            continue
        new_tasks.append(task)
    taskset = TaskSet(new_tasks)

    pending_tasks = []
    number_of_tasks_with_different_input_output_size = 0
    for task in taskset.tasks:
        if task.has_same_input_output_size_for_all_examples():
            pending_tasks.append(task)
        else:
            number_of_tasks_with_different_input_output_size += 1
    
    random.Random(seed + 2).shuffle(pending_tasks)
    # truncate the list to a few tasks
    # pending_tasks = pending_tasks[:20]
    pending_tasks = sorted(pending_tasks, key=lambda task: task.metadata_task_id)
    # print(f"Number of tasks with different input/output size: {number_of_tasks_with_different_input_output_size}")
    # print(f"Number of tasks with same input/output size: {len(pending_tasks)}")
    print(f"After filtering, number of tasks in group '{groupname}': {len(pending_tasks)}")
    datasetid_groupname_task_list.append((dataset_id, groupname, pending_tasks))

for combo_index, combo in enumerate(featurecomboitem_list):
    print(f"Feature combo {combo_index+1} of {len(featurecomboitem_list)}, features: {combo.feature_names_sorted()}")
    save_dir = f'run_tasks_result/measure_feature_combinations/{run_id}/{combo.run_index}'
    jsonl_filepath = f'{save_dir}/results.jsonl'
    summary_filepath = f'{save_dir}/summary.json'

    job_list = []
    for (dataset_id, groupname, tasks) in datasetid_groupname_task_list:
        for task in tasks:
            job_list.append((dataset_id, groupname, task))

    feature_name_list = combo.feature_names_sorted()

    correct_count = 0
    total_elapsed_float = 0
    count_score9599 = 0
    count_score9094 = 0
    with tqdm(job_list, desc="Processing tasks", leave=False, position=0) as pbar:
        for (dataset_id, groupname, task) in pbar:
            print(f"Processing task {task.metadata_task_id}, dataset '{dataset_id}', group '{groupname}'")

            desc = task.metadata_task_id
            # truncate string to 20 characters, if it's longer add ...
            desc = (desc[:20] + '...') if len(desc) > 20 else desc
            pbar.set_description(desc)

            for test_index in range(task.count_tests):
                current_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                
                start_time = time.perf_counter()

                predicted_output_result = ModelBeta1.predict_output(
                    task, 
                    test_index, 
                    previous_prediction_image=None,
                    previous_prediction_mask=None,
                    refinement_index=0, 
                    noise_level=100,
                    features=combo.features,
                )
                predicted_output = predicted_output_result.images(1)[0]

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

                if is_correct == False:
                    if track_incorrect_prediction is not None:
                        features_str = ImageFeature.names_sorted_and_joined(combo.features, separator=',')
                        metadata = f'run={run_id} measure_feature_combinations solver=beta1 features={features_str}'
                        track_incorrect_prediction.track_incorrect_prediction_with_raw_data(
                            dataset_id=dataset_id, 
                            task_id=task.metadata_task_id, 
                            test_index=test_index, 
                            test_input=input_image, 
                            test_output=expected_output_image, 
                            predicted_output=predicted_output, 
                            metadata=metadata
                        )

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
                    "version": "simon_arc_lab measure_feature_combinations 2025-jan-08"
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
