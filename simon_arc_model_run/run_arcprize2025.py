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

def create_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run1(production: bool, run_id: str, input_challenges_path: str, save_debug_dir: str, output_submission_path: str):
    # print the parameters
    mode_name = 'production' if production else 'developer'
    print(f"Mode: {mode_name}")
    print(f"Run id: {run_id}")
    print(f"Debug data will be saved to '{save_debug_dir}'")
    print(f"Submission will be saved to '{output_submission_path}'")
    print(f"Gallery will be saved to '{save_debug_dir}'")

    dataset_id = 'ARC-AGI-2-test'

    taskset = TaskSet.load_kaggle_arcprize2024_json(input_challenges_path)
    if not production:
        taskset.keep_tasks_with_id(set(['00576224', '009d5c81', '00d62c1b']), verbose=False)

    wm = WorkManagerDecisionTree(run_id=run_id, dataset_id=dataset_id, taskset=taskset, cache_dir=None, incorrect_predictions_jsonl_path=None)
    # wm.truncate_work_items(20)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_debug_dir)
    # wm.process_all_work_items(show=True)
    wm.discard_items_where_predicted_output_is_identical_to_the_input()
    wm.summary()

    gallery_title=f"arcprize2025-test, {run_id}"
    gallery_generator_run(save_debug_dir, title=gallery_title)

    # Save the submission
    wm.save_arcprize2024_submission_file(output_submission_path)

if __name__ == '__main__':
    run_id: str = create_run_id()
    run1(
        production=False,
        run_id=run_id,
        input_challenges_path='testdata/kaggle-arc-prize-2025/arc-agi_test_challenges.json', 
        save_debug_dir=f'run_tasks_result/{run_id}',
        output_submission_path=f'run_tasks_result/{run_id}/submission.json'
    )
