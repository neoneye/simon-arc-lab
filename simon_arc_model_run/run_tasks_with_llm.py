from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_model.model_alpha1 import ModelAlpha1
from simon_arc_model.work_manager_simple import WorkManagerSimple

# A power of 2 value, and the max length of the input prompt
CONTEXT_SIZE_LIMIT = (512, 500)
# CONTEXT_SIZE_LIMIT = (1024, 1000)

# model_iteration, predictor_id = (471, 'v1')
model_iteration, predictor_id = (625, 'v1')
# model_iteration, predictor_id = (768, 'v3')
model_name = f'simon-arc-lab-model{model_iteration}'
model_directory = f'/Users/neoneye/nobackup/git/{model_name}'

# check if the model is a dir in the file system
if not os.path.isdir(model_directory):
    print(f"Model directory '{model_directory}' does not exist.")
    sys.exit(1)

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

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

context_length, max_prompt_length = CONTEXT_SIZE_LIMIT
print(f"context length: {context_length}, max prompt length: {max_prompt_length}")

# Load model
model = ModelAlpha1(model_directory, context_length)

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

# incorrect_predictions_jsonl_path = 'run_tasks_result/incorrect_prediction.jsonl'
# incorrect_predictions_jsonl_path = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
incorrect_predictions_jsonl_path = None

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)
    # taskset.remove_tasks_by_id(taskids_to_ignore, verbose=False)

    wm = WorkManagerSimple(run_id, dataset_id, model, taskset, model_name, predictor_id, incorrect_predictions_jsonl_path)
    # wm.discard_items_with_too_short_prompts(500)
    wm.discard_items_with_too_long_prompts(max_prompt_length)
    # wm.truncate_work_items(50)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_dir)
    # wm.process_all_work_items(show=True)
    wm.discard_items_where_predicted_output_is_identical_to_the_input()
    wm.summary()

    gallery_title = f'{groupname}, model {model_iteration}'
    gallery_generator_run(save_dir, title=gallery_title)

    # wm.save_arcprize2024_submission_file('submission.json')
