import os
import json
from simon_arc_lab.taskset import TaskSet
from .work_item import WorkItem

def collect_predictions_as_arcprize2024_submission_dict(taskset: TaskSet, work_items: list[WorkItem]) -> dict:
    result_dict = {}
    # Kaggle requires all the original tasks to be present in the submission file.
    # Create empty entries in the result_dict, with empty images, with the correct number of test pairs.
    for task in taskset.tasks:
        task_id = task.metadata_task_id
        count_tests = task.count_tests
        empty_image = []
        attempts_dict = {
            'attempt_1': empty_image,
            'attempt_2': empty_image,
        }
        test_list = []
        for _ in range(count_tests):
            test_list.append(attempts_dict)
        result_dict[task_id] = test_list
            
    # Update the result_dict with the predicted images
    attempt1_set = set()
    for work_item in work_items:
        if work_item.predicted_output_image is None:
            continue
        task_id = work_item.task.metadata_task_id

        if task_id in attempt1_set:
            attempt_1or2 = 'attempt_2'
            # print(f'Found multiple predictions for task {task_id}. Using attempt_2.')
        else:
            attempt_1or2 = 'attempt_1'
            attempt1_set.add(task_id)

        # Update the existing entry in the result_dict with the predicted image
        image = work_item.predicted_output_image.tolist()
        result_dict[task_id][work_item.test_index][attempt_1or2] = image
    return result_dict

def save_arcprize2024_submission_file(path_to_json_file: str, json_dict: dict):
    with open(path_to_json_file, 'w') as f:
        json.dump(json_dict, f)
    file_size = os.path.getsize(path_to_json_file)
    print(f"Wrote {file_size} bytes to file: {path_to_json_file}")
