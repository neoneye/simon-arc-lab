from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_model.model import Model
from simon_arc_model.work_manager_decision_tree import WorkManagerDecisionTree

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

cache_dir = 'run_tasks_result/cache_decisiontree'
os.makedirs(cache_dir, exist_ok=True)
cache_dir = None

incorrect_predictions_jsonl_path = 'run_tasks_result/incorrect_predictions.jsonl'

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)
    # taskset.remove_tasks_by_id(taskids_to_ignore, verbose=False)

    wm = WorkManagerDecisionTree(run_id, dataset_id, taskset, cache_dir, incorrect_predictions_jsonl_path)
    # wm.discard_items_with_too_short_prompts(500)
    # wm.discard_items_with_too_long_prompts(max_prompt_length)
    # wm.truncate_work_items(20)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_dir)
    # wm.process_all_work_items(show=True)
    wm.discard_items_where_predicted_output_is_identical_to_the_input()
    wm.summary()

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_run(save_dir, title=gallery_title)

    # wm.save_arcprize2024_submission_file('submission.json')
