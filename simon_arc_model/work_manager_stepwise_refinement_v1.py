import os
from tqdm import tqdm
import numpy as np
from typing import Optional
from simon_arc_lab.rle.deserialize import DeserializeError
from simon_arc_lab.image_distort import *
from simon_arc_lab.image_noise import *
from simon_arc_lab.image_vote import *
from simon_arc_lab.task import Task
from simon_arc_lab.task_mutator import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_prediction_result
from .model_alpha1 import ModelAlpha1, ModelAlpha1ProcessMode
from .predict_output_v2 import *
from .work_item_with_predictor import WorkItemWithPredictor
from .work_item_list import WorkItemList
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *
from .work_manager_base import WorkManagerBase

class WorkManagerStepwiseRefinementV1(WorkManagerBase):
    def __init__(self, model: ModelAlpha1, taskset: TaskSet):
        self.model = model
        self.taskset = taskset
        self.work_items = WorkManagerStepwiseRefinementV1.create_work_items(taskset)
        self.work_items_finished = []

    @classmethod
    def create_work_items(cls, taskset: TaskSet) -> list['WorkItem']:
        task_mutator_class_list = [TaskMutatorOriginal]
        # task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose]
        # task_mutator_class_list = [TaskMutatorOriginal, TaskMutatorTranspose, TaskMutatorInputRotateCW, TaskMutatorInputRotateCCW, TaskMutatorInputRotate180, TaskMutatorTransposeSoInputIsMostCompact]
        refinement_step = 0
        work_items = []
        for task in taskset.tasks:
            for test_index in range(task.count_tests):
                for task_mutator_class in task_mutator_class_list:
                    previous_predicted_image = None
                    predictor = PredictOutputV2(task, test_index, task_mutator_class, previous_predicted_image)
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
        self.work_items_finished = WorkItemList.discard_items_where_predicted_output_is_identical_to_the_input(self.work_items_finished)
    
    def process_all_work_items(self, show: bool = False, save_dir: Optional[str] = None):
        if save_dir is not None:
            print(f'Saving images to directory: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

        refinement_mode_list = [
            ModelAlpha1ProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_HIGH,
            # ModelProcessMode.TEMPERATURE_MEDIUM,
            ModelAlpha1ProcessMode.TEMPERATURE_LAB1,
            ModelAlpha1ProcessMode.TEMPERATURE_LAB1,
            ModelAlpha1ProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_LAB1,
            # ModelProcessMode.TEMPERATURE_HIGH,
            # ModelProcessMode.TEMPERATURE_HIGH,
            # ModelProcessMode.TEMPERATURE_HIGH,
            # ModelProcessMode.TEMPERATURE_ZERO_BEAM5,
            # ModelProcessMode.TEMPERATURE_MEDIUM,
            # ModelProcessMode.TEMPERATURE_MEDIUM,
            # ModelProcessMode.TEMPERATURE_LOW,
            # ModelProcessMode.TEMPERATURE_LOW,
            # ModelProcessMode.TEMPERATURE_ZERO_BEAM5,
            # ModelProcessMode.TEMPERATURE_MEDIUM,
            # ModelProcessMode.TEMPERATURE_LOW,
            # ModelProcessMode.TEMPERATURE_ZERO_BEAM5,
            # ModelProcessMode.TEMPERATURE_MEDIUM,
            # ModelProcessMode.TEMPERATURE_LOW,
            # ModelProcessMode.TEMPERATURE_ZERO_BEAM5,
        ]
        number_of_refinement_steps = len(refinement_mode_list)
        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        self.work_items_finished = []
        for original_work_item in pbar:
            work_item = original_work_item

            predicted_images = []
            for refinement_step in range(number_of_refinement_steps):
                # if refinement_step == number_of_refinement_steps - 1:
                #     mode = ModelProcessMode.TEMPERATURE_ZERO_BEAM5
                # elif refinement_step > 2:
                #     mode = ModelProcessMode.TEMPERATURE_MEDIUM
                # else:
                #     mode = ModelProcessMode.TEMPERATURE_HIGH
                mode = refinement_mode_list[refinement_step]
                context = {
                    'model': self.model,
                    'mode': mode,
                }
                work_item.process(context)
                self.work_items_finished.append(work_item)

                if work_item.status == WorkItemStatus.CORRECT:
                    correct_task_id_set.add(original_work_item.task.metadata_task_id)
                    correct_count = len(correct_task_id_set)
                pbar.set_postfix({'correct': correct_count})
                if show:
                    work_item.show()
                if save_dir is not None:
                    work_item.show(save_dir)

                if work_item.status == WorkItemStatus.PROBLEM_DESERIALIZE:
                    break
                if work_item.status == WorkItemStatus.PROBLEM_MISSING_PREDICTION_IMAGE:
                    break
                
                new_task = original_work_item.task.clone()
                # image_index = new_task.count_examples + original_work_item.test_index
                # new_task.output_images[image_index] = work_item.predicted_output_image

                predicted_images.append(work_item.predicted_output_image.copy())

                # IDEA: pick a random mutator
                if refinement_step % 2 == 0:
                    task_mutator_class = TaskMutatorTranspose
                    previous_predicted_image = np.transpose(work_item.predicted_output_image)
                else:
                    task_mutator_class = TaskMutatorOriginal
                    previous_predicted_image = work_item.predicted_output_image
                # task_mutator_class = TaskMutatorOriginal
                # previous_predicted_image = work_item.predicted_output_image

                iteration_seed = 42 + refinement_step
                previous_predicted_image = image_distort(previous_predicted_image, 1, 10, iteration_seed + 1)
                previous_predicted_image = image_noise_one_pixel(previous_predicted_image, iteration_seed + 2)
                predictor = PredictOutputV2(new_task, work_item.test_index, task_mutator_class, previous_predicted_image)
                next_work_item = WorkItemWithPredictor(new_task, work_item.test_index, refinement_step+1, predictor)
                work_item = next_work_item

            try:
                the_image = image_vote(predicted_images)
            except ValueError as e:
                print(f'Error in image_vote: {e}')
                the_image = None

            if the_image is not None:
                if np.array_equal(the_image, original_work_item.task.test_output(original_work_item.test_index)):
                    status = 'correct'
                else:
                    status = 'incorrect'
                WorkManagerStepwiseRefinementV1.show_voted_image(original_work_item.task, original_work_item.test_index, the_image, status, save_dir)

    @classmethod
    def show_voted_image(cls, task: Task, test_index: int, predicted_output_image: np.array, status_string: str, save_dir_path: Optional[str]):
        input_image = task.test_input(test_index)
        task_id = task.metadata_task_id
        title = f'{task_id} test={test_index} vote {status_string}'

        expected_output_image = task.test_output(test_index)

        filename = f'{task_id}_test{test_index}_vote_{status_string}.png'
        if save_dir_path is not None:
            save_path = os.path.join(save_dir_path, filename)
        else:
            save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)

    def summary(self):
        correct_task_id_set = set()
        for work_item in self.work_items_finished:
            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
        correct_count = len(correct_task_id_set)
        print(f'Number of correct solutions: {correct_count}')

        counters = {}
        for work_item in self.work_items_finished:
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
        # IDEA: Extract the best predictions from work_items_finished. How to determine the best?
        json_dict = collect_predictions_as_arcprize2024_submission_dict(self.taskset, self.work_items_finished)
        save_arcprize2024_submission_file(path_to_json_file, json_dict)
