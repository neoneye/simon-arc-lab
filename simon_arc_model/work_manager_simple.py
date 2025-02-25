import os
from tqdm import tqdm
from typing import Optional
from simon_arc_lab.rle.deserialize import DeserializeError
from simon_arc_lab.task_mutator import *
from simon_arc_lab.taskset import TaskSet
from .model_alpha1 import ModelAlpha1
from .predict_output_v1 import *
from .predict_output_v3 import *
from .work_item_with_predictor import WorkItemWithPredictor
from .work_item_list import WorkItemList
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *
from .work_manager_base import WorkManagerBase
from .track_incorrect_prediction import TrackIncorrectPrediction

class WorkManagerSimple(WorkManagerBase):
    def __init__(self, run_id: str, dataset_id: str, model: ModelAlpha1, taskset: TaskSet, model_name: str, predictor_id: str, incorrect_predictions_jsonl_path: Optional[str] = None):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.model = model
        self.taskset = taskset
        self.predictor_id = predictor_id
        self.work_items = WorkManagerSimple.create_work_items(taskset, predictor_id)
        self.model_name = model_name
        self.incorrect_predictions_jsonl_path = incorrect_predictions_jsonl_path

    @classmethod
    def create_work_items(cls, taskset: TaskSet, predictor_id: str) -> list['WorkItem']:
        task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose]
        refinement_step = None
        work_items = []
        for task in taskset.tasks:
            for test_index in range(task.count_tests):
                for task_mutator_class in task_mutator_class_list:
                    if predictor_id == 'v1':
                        predictor = PredictOutputV1(task, test_index, task_mutator_class)
                    elif predictor_id == 'v3':
                        predictor = PredictOutputV3(task, test_index, task_mutator_class)
                    else:
                        raise ValueError(f'Unknown predictor_id: {predictor_id}')
                    work_item = WorkItemWithPredictor(task, test_index, refinement_step, predictor)
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

        # Track incorrect predictions
        if self.predictor_id == 'v1':
            model_description = 'CodeT5_RLE_input_and_RLE_output'
        elif self.predictor_id == 'v3':
            model_description = 'CodeT5_RLE_input_and_pixel_output'
        else:
            raise ValueError(f'Unknown predictor_id: {self.predictor_id}')
        incorrect_prediction_metadata = f'run={self.run_id} solver={self.model_name}_{model_description}'
        incorrect_prediction_dataset_id = self.dataset_id
        if self.incorrect_predictions_jsonl_path is not None:
            track_incorrect_prediction = TrackIncorrectPrediction.load_from_jsonl(self.incorrect_predictions_jsonl_path)
        else:
            track_incorrect_prediction = None

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

            if track_incorrect_prediction is not None:
                track_incorrect_prediction.track_incorrect_prediction_with_workitem(
                    work_item,
                    incorrect_prediction_dataset_id, 
                    work_item.predicted_output_image,
                    incorrect_prediction_metadata
                )

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
