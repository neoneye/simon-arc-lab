from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_model.work_manager_stepwise_refinement_v4 import WorkManagerStepwiseRefinementV4
from simon_arc_model.arc_bad_prediction import *
from simon_arc_model.work_item_with_previousprediction import WorkItemWithPreviousPrediction

max_number_of_bad_predictions_per_task = 4

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

arc_bad_prediction_file = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
arc_bad_prediction_dataset = ARCBadPredictionDataset.load(arc_bad_prediction_file)
# arc_bad_prediction_dataset.display_sample_records()

dataset_task = set()
for record in arc_bad_prediction_dataset.records:
    dataset_id = record.dataset
    task_id = record.task
    dataset_task.add((dataset_id, task_id))

print(f"Number of unique tasks in the arc-bad-prediction dataset: {len(dataset_task)}")

incorrect_predictions_jsonl_path = arc_bad_prediction_file
#incorrect_predictions_jsonl_path = None

task_ids_of_interest_list = [
    # '08573cc6',
    # 'e5c44e8f',
    # '5c2c9af4',
    # 'f8c80d96',
    # '28e73c20',
    # '09c534e7',
    # '05a7bcf2_v2',
    # '1c02dbbe',
    # '1e97544e',
    # '292dd178',
    # '29700607',
    # '40f6cd08',
    # '3f23242b',
    # '5b526a93',
    'ecdecbb3',
    'db93a21d',
    'd06dbe63',
    '928ad970',
    '7df24a62',
    '673ef223',
    'e619ca6e',
    '58e15b12_v3',
    'da515329',
]
task_ids_of_interest = set(task_ids_of_interest_list)


number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    # Keep only that are present in the arc-bad-prediction dataset. Remove the rest.
    task_ids_to_ignore = set()
    for task in taskset.tasks:
        task_id = task.metadata_task_id
        key = (dataset_id, task_id)
        found = key in dataset_task
        if not found:
            task_ids_to_ignore.add(task_id)
    taskset.remove_tasks_by_id(taskids_to_ignore, verbose=False)

    taskset.keep_tasks_with_id(task_ids_of_interest, verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    # Create work items for each bad prediction
    work_items = []

    count_key_occurences = dict()
    for task in taskset.tasks:
        if task.has_same_input_output_size_for_all_examples() == False:
            continue
        task_id = task.metadata_task_id
        find_key = (dataset_id, task_id)
        for record in arc_bad_prediction_dataset.records:
            record_dataset_id = record.dataset
            record_task_id = record.task
            record_key = (record_dataset_id, record_task_id)
            if find_key != record_key:
                continue

            test_index = record.test_index
            if test_index >= task.count_tests:
                print(f"Skipping task: {task_id}, due to test index {test_index} is out of range.")
                continue

            # Only take the first N bad predictions for each task. Ignore the rest of the bad predictions.
            process_key = (dataset_id, task_id, test_index)
            count = count_key_occurences.get(process_key, 0)
            count += 1
            count_key_occurences[process_key] = count
            if count > max_number_of_bad_predictions_per_task:
                continue

            unique_id = str(record.line_number)
            work_item = WorkItemWithPreviousPrediction(task, test_index, record.predicted_output, unique_id)
            work_items.append(work_item)

    print(f"Number of work items: {len(work_items)}")

    wm = WorkManagerStepwiseRefinementV4(run_id, dataset_id, taskset, work_items, incorrect_predictions_jsonl_path)
    # wm.truncate_work_items(40)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_dir)
    # wm.process_all_work_items(show=True)
    wm.discard_items_where_predicted_output_is_identical_to_the_input()
    wm.summary()

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_run(save_dir, title=gallery_title)

    # wm.save_arcprize2024_submission_file('submission.json')
