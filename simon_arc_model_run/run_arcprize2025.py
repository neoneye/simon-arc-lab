"""
USAGE:
(venv) PROMPT> python simon_arc_model_run/run_arcprize2025.py
"""
from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_model.work_manager_decision_tree import WorkManagerDecisionTree

def run1(production: bool=False):
    mode_name = 'production' if production else 'developer'
    print(f"Mode: {mode_name}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run id: {run_id}")

    dataset_id = 'ARC-AGI-2-test'
    groupname = 'arcprize2025-test'

    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2025/arc-agi_test_challenges.json')
    if not production:
        taskset.keep_tasks_with_id(set(['00576224', '009d5c81', '00d62c1b']), verbose=False)

    wm = WorkManagerDecisionTree(run_id=run_id, dataset_id=dataset_id, taskset=taskset, cache_dir=None, incorrect_predictions_jsonl_path=None)
    # wm.truncate_work_items(20)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_dir)
    # wm.process_all_work_items(show=True)
    wm.discard_items_where_predicted_output_is_identical_to_the_input()
    wm.summary()

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_run(save_dir, title=gallery_title)

    # wm.save_arcprize2024_submission_file('submission.json')

if __name__ == '__main__':
    run1(production=False)
