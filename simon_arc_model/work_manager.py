import os
import sys
from tqdm import tqdm
import json
from enum import Enum
import numpy as np
from typing import Optional

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_prediction_result
from .model import Model
from .predict_output_v1 import PredictOutputV1

class WorkItemStatus(Enum):
    UNASSIGNED = 0
    UNVERIFIED = 1
    CORRECT = 2
    INCORRECT = 3
    PROBLEM_MISSING_PREDICTION_IMAGE = 4
    PROBLEM_DESERIALIZE = 5

    def to_string(self):
        if self == WorkItemStatus.PROBLEM_DESERIALIZE:
            return 'problemdeserialize'
        if self == WorkItemStatus.PROBLEM_MISSING_PREDICTION_IMAGE:
            return 'problemmissingpredictionimage'
        return self.name.lower()

class WorkItem:
    def __init__(self, task: Task, test_index: int):
        self.task = task
        self.test_index = test_index
        self.predictor = PredictOutputV1(task, test_index)
        self.predicted_output_image = None
        self.status = WorkItemStatus.UNASSIGNED

    def process(self, model: Model):
        self.predictor.execute(model)

        task = self.task
        test_index = self.test_index
        task_id = task.metadata_task_id
        expected_output_image = task.test_output(test_index)

        try:
            self.predicted_output_image = self.predictor.predicted_image()
        except Exception as e:
            print(f'Error deserializing response for task {task_id} test={test_index}. Error: {e}')
            self.status = WorkItemStatus.PROBLEM_DESERIALIZE
            return

        if self.predicted_output_image is None:
            self.status = WorkItemStatus.PROBLEM_MISSING_PREDICTION_IMAGE
            return
        
        if expected_output_image is None:
            self.status = WorkItemStatus.UNVERIFIED
            return
        
        if np.array_equal(self.predicted_output_image, expected_output_image):
            self.status = WorkItemStatus.CORRECT
        else:
            self.status = WorkItemStatus.INCORRECT

    def show(self, save_dir_path: Optional[str] = None):
        task = self.task
        test_index = self.test_index
        input_image = task.test_input(test_index)
        task_id = task.metadata_task_id
        status_string = self.status.to_string()
        title = f'{task_id} test={test_index} {status_string}'

        expected_output_image = task.test_output(test_index)
        predicted_output_image = self.predicted_output_image

        filename = f'{task_id}_test{test_index}_{status_string}.png'
        if save_dir_path is not None:
            save_path = os.path.join(save_dir_path, filename)
        else:
            save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)


class WorkManager:
    def __init__(self, model: Model, taskset: TaskSet):
        self.model = model
        self.taskset = taskset
        self.work_items = WorkManager.create_work_items(taskset)

    @classmethod
    def create_work_items(cls, taskset: TaskSet) -> list['WorkItem']:
        work_items = []
        for task in taskset.tasks:
            for test_index in range(task.count_tests):
                work_item = WorkItem(task, test_index)
                work_items.append(work_item)
        return work_items

    def discard_items_with_too_long_prompts(self, max_prompt_length: int):
        """
        Ignore those where the prompt longer than what the model can handle.
        """
        count_before = len(self.work_items)
        filtered_work_items = []
        for work_item in self.work_items:
            if len(work_item.predictor.prompt()) <= max_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        self.work_items = filtered_work_items
        print(f'Removed {count_before - count_after} work items with too long prompt. Remaining are {count_after} work items.')

    def process_all_work_items(self, show: bool = False, save_dir: Optional[str] = None):
        if save_dir is not None:
            print(f'Saving images to directory: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

        correct_count = 0
        pbar = tqdm(self.work_items, desc="Processing work items")
        for work_item in pbar:
            work_item.process(self.model)
            if work_item.status == WorkItemStatus.CORRECT:
                correct_count += 1
            pbar.set_postfix({'correct': correct_count})
            if show:
                work_item.show()
            if save_dir is not None:
                work_item.show(save_dir)

    def summary(self):
        counters = {}
        for work_item in self.work_items:
            status = work_item.status
            if status in counters:
                counters[status] += 1
            else:
                counters[status] = 1
        for status, count in counters.items():
            print(f'{status.name}: {count}')

    def legacy_collect_predictions_as_arcprize2024_submission_dict(self) -> dict:
        result_dict = {}
        for work_item in self.work_items:
            if work_item.predicted_output_image is None:
                continue
            task_id = work_item.task.metadata_task_id

            # Create a new entry in the result_dict if it doesn't exist, with dummy images
            # This is in order to handle tasks that have 2 or more test pairs.
            if task_id not in result_dict:
                count_tests = work_item.task.count_tests
                empty_image = []
                attempts_dict = {
                    'attempt_1': empty_image,
                    'attempt_2': empty_image,
                }
                test_list = []
                for _ in range(count_tests):
                    test_list.append(attempts_dict)
                result_dict[task_id] = test_list

            # Update the existing entry in the result_dict with the predicted image
            image = work_item.predicted_output_image.tolist()
            result_dict[task_id][work_item.test_index]['attempt_1'] = image
        return result_dict
    
    def collect_predictions_as_arcprize2024_submission_dict(self) -> dict:
        result_dict = {}
        # My hypothesis is that Kaggle requires all tasks to be present in the submission? 
        # Using all the original tasks from the original taskset. 
        # Create empty entries in the result_dict, with empty images, with the correct number of test pairs.
        for task in self.taskset.tasks:
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
        for work_item in self.work_items:
            if work_item.predicted_output_image is None:
                continue
            task_id = work_item.task.metadata_task_id

            # Update the existing entry in the result_dict with the predicted image
            image = work_item.predicted_output_image.tolist()
            result_dict[task_id][work_item.test_index]['attempt_1'] = image
        return result_dict
    
    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        dict = self.collect_predictions_as_arcprize2024_submission_dict()
        with open(path_to_json_file, 'w') as f:
            json.dump(dict, f)
        file_size = os.path.getsize(path_to_json_file)
        print(f"Wrote {file_size} bytes to file: {path_to_json_file}")
