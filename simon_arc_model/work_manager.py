import os
import sys
from tqdm import tqdm
import json
from enum import Enum
import numpy as np
from typing import Optional
from simon_arc_lab.rle.deserialize import DeserializeError
from simon_arc_lab.task import Task
from simon_arc_lab.task_mutator import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_prediction_result
from .model import Model
from .predict_output_v1 import *

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
    def __init__(self, task: Task, test_index: int, predictor: PredictOutputBase):
        self.predictor_name = predictor.name()
        # print(f'WorkItem: task={task.metadata_task_id} test={test_index} predictor={self.predictor_name}')
        self.task = task
        self.test_index = test_index
        self.predictor = predictor
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
        except DeserializeError as e:
            # print(f'RLE decoding error for task {task_id} test={test_index}. Error: {e} score={e.score}')
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
        title = f'{task_id} test={test_index} {self.predictor_name} {status_string}'

        expected_output_image = task.test_output(test_index)
        predicted_output_image = self.predicted_output_image

        filename = f'{task_id}_test{test_index}_{self.predictor_name}_{status_string}.png'
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
        task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose]
        # task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose, TaskMutatorInputRotateCW, TaskMutatorInputRotateCCW, TaskMutatorInputRotate180, TaskMutatorTransposeSoInputIsMostCompact]
        work_items = []
        for task in taskset.tasks:
            for test_index in range(task.count_tests):
                for task_mutator_class in task_mutator_class_list:
                    predictor = PredictOutputV1(task, test_index, task_mutator_class)
                    work_item = WorkItem(task, test_index, predictor)
                    try:
                        prompt = work_item.predictor.prompt()
                        work_items.append(work_item)
                    except DeserializeError as e:
                        print(f'Error cannot construct prompt for {task.metadata_task_id} test={test_index}. Error: {e}')
        return work_items

    def truncate_work_items(self, max_count: int):
        self.work_items = self.work_items[:max_count]

    def discard_items_with_too_long_prompts(self, max_prompt_length: int):
        """
        Ignore those where the prompt longer than what the model can handle.
        """
        count_before = len(self.work_items)
        filtered_work_items = []
        for work_item in self.work_items:
            prompt_length = len(work_item.predictor.prompt())
            if prompt_length <= max_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        self.work_items = filtered_work_items
        print(f'Removed {count_before - count_after} work items with too long prompt. Remaining are {count_after} work items.')

    def discard_items_with_too_short_prompts(self, min_prompt_length: int):
        """
        Ignore those where the prompt shorter than N tokens.
        """
        count_before = len(self.work_items)
        filtered_work_items = []
        for work_item in self.work_items:
            prompt_length = len(work_item.predictor.prompt())
            if prompt_length >= min_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        self.work_items = filtered_work_items
        print(f'Removed {count_before - count_after} work items with too short prompt. Remaining are {count_after} work items.')

    def process_all_work_items(self, show: bool = False, save_dir: Optional[str] = None):
        if save_dir is not None:
            print(f'Saving images to directory: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        for work_item in pbar:
            work_item.process(self.model)
            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
                correct_count = len(correct_task_id_set)
            pbar.set_postfix({'correct': correct_count})
            if show:
                work_item.show()
            if save_dir is not None:
                work_item.show(save_dir)

    def summary(self):
        correct_task_id_set = set()
        for work_item in self.work_items:
            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
        correct_count = len(correct_task_id_set)
        print(f'Number of correct solutions: {correct_count}')

        counters = {}
        for work_item in self.work_items:
            predictor_name = work_item.predictor_name
            status_name = work_item.status.name
            key = f'{predictor_name}_{status_name}'
            if key in counters:
                counters[key] += 1
            else:
                counters[key] = 1
        for key, count in counters.items():
            print(f'{key}: {count}')

    def discard_items_where_predicted_output_is_identical_to_the_input(self):
        """
        Usually in ARC-AGI the predicted output image is supposed to be different from the input image.
        There are ARC like datasets where the input and output may be the same, but it's rare.
        It's likely a mistake when input and output is the same.
        """
        count_before = len(self.work_items)
        filtered_work_items = []
        for work_item in self.work_items:
            if work_item.predicted_output_image is None:
                filtered_work_items.append(work_item)
                continue
            input_image = work_item.task.test_input(work_item.test_index)
            predicted_image = work_item.predicted_output_image
            is_identical = np.array_equal(input_image, predicted_image)
            if not is_identical:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        self.work_items = filtered_work_items
        print(f'Removed {count_before - count_after} work items where the input and output is identical. Remaining are {count_after} work items.')

    def collect_predictions_as_arcprize2024_submission_dict(self) -> dict:
        result_dict = {}
        # Kaggle requires all the original tasks to be present in the submission file.
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
        attempt1_set = set()
        for work_item in self.work_items:
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
    
    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        dict = self.collect_predictions_as_arcprize2024_submission_dict()
        with open(path_to_json_file, 'w') as f:
            json.dump(dict, f)
        file_size = os.path.getsize(path_to_json_file)
        print(f"Wrote {file_size} bytes to file: {path_to_json_file}")
