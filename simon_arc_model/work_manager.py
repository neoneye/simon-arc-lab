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
from .model import Model
from .predict_output_v1 import *
from .work_item import WorkItem
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *

class WorkManager:
    def __init__(self, model: Model, taskset: TaskSet):
        self.model = model
        self.taskset = taskset
        self.work_items = WorkManager.create_work_items(taskset)

    @classmethod
    def create_work_items(cls, taskset: TaskSet) -> list['WorkItem']:
        task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose]
        # task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose, TaskMutatorInputRotateCW, TaskMutatorInputRotateCCW, TaskMutatorInputRotate180, TaskMutatorTransposeSoInputIsMostCompact]
        refinement_step = None
        work_items = []
        for task in taskset.tasks:
            for test_index in range(task.count_tests):
                for task_mutator_class in task_mutator_class_list:
                    predictor = PredictOutputV1(task, test_index, task_mutator_class)
                    work_item = WorkItem(task, test_index, refinement_step, predictor)
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
            context = {
                'model': self.model,
            }
            work_item.process(context)
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
    
    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        json_dict = collect_predictions_as_arcprize2024_submission_dict(self.taskset, self.work_items)
        save_arcprize2024_submission_file(path_to_json_file, json_dict)
