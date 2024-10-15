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
from .work_item_list import WorkItemList
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
        self.work_items = WorkItemList.discard_items_with_too_long_prompts(self.work_items, max_prompt_length)

    def discard_items_with_too_short_prompts(self, min_prompt_length: int):
        self.work_items = WorkItemList.discard_items_with_too_short_prompts(self.work_items, min_prompt_length)

    def discard_items_where_predicted_output_is_identical_to_the_input(self):
        self.work_items = WorkItemList.discard_items_where_predicted_output_is_identical_to_the_input(self.work_items)
    
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

    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        json_dict = collect_predictions_as_arcprize2024_submission_dict(self.taskset, self.work_items)
        save_arcprize2024_submission_file(path_to_json_file, json_dict)
