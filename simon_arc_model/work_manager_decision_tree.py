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
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result
from .predict_output_donothing import PredictOutputDoNothing
from .work_item import WorkItem
from .work_item_list import WorkItemList
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *
from .work_manager_base import WorkManagerBase
from .decision_tree_util import DecisionTreeUtil, DecisionTreeFeature

# Correct 59, Solves 1 of the hidden ARC tasks
# ARC-AGI training=41, evaluation=17
FEATURES_1 = [
    DecisionTreeFeature.COMPONENT_NEAREST4,
    DecisionTreeFeature.HISTOGRAM_DIAGONAL,
    DecisionTreeFeature.HISTOGRAM_ROWCOL,
    DecisionTreeFeature.HISTOGRAM_VALUE,
    DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL,
    DecisionTreeFeature.BOUNDING_BOXES,
]

# Correct 59, Solves 1 of the hidden ARC tasks
# ARC-AGI training=39, evaluation=20
FEATURES_2 = [
    DecisionTreeFeature.BOUNDING_BOXES,
    DecisionTreeFeature.COMPONENT_NEAREST4,
    DecisionTreeFeature.EROSION_ALL8,
    DecisionTreeFeature.HISTOGRAM_ROWCOL,
]

class WorkManagerDecisionTree(WorkManagerBase):
    def __init__(self, model: any, taskset: TaskSet):
        self.taskset = taskset
        self.work_items = WorkManagerDecisionTree.create_work_items(taskset)

    @classmethod
    def create_work_items(cls, taskset: TaskSet) -> list['WorkItem']:
        work_items = []
        for task in taskset.tasks:
            if DecisionTreeUtil.has_same_input_output_size_for_all_examples(task) == False:
                continue

            for test_index in range(task.count_tests):
                work_item = WorkItem(task, test_index, None, PredictOutputDoNothing())
                work_items.append(work_item)
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

        # noise_levels = [95, 90, 85, 80, 75, 70, 65]
        # noise_levels = [95, 90]
        # noise_levels = [100, 95, 90]
        noise_levels = [100]
        number_of_refinements = len(noise_levels)

        features = set(FEATURES_2)

        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        for original_work_item in pbar:
            work_item = original_work_item

            ts = TaskSimilarity.create_with_task(work_item.task)

            image_and_score = []

            last_predicted_output = None
            for refinement_index in range(number_of_refinements):
                noise_level = noise_levels[refinement_index]
                # print(f"Refinement {refinement_index+1}/{number_of_refinements} noise_level={noise_level}")
                predicted_output = DecisionTreeUtil.predict_output(
                    work_item.task, 
                    work_item.test_index, 
                    last_predicted_output, 
                    refinement_index, 
                    noise_level,
                    features
                )
                last_predicted_output = predicted_output
                score = ts.measure_test_prediction(predicted_output, work_item.test_index)
                # print(f"task: {work_item.task.metadata_task_id} score: {score} refinement_index: {refinement_index} noise_level: {noise_level}")
                image_and_score.append((predicted_output, score))

            best_image, best_score = max(image_and_score, key=lambda x: x[1])
            # print(f"task: {work_item.task.metadata_task_id} best_score: {best_score}")

            work_item.predicted_output_image = best_image
            work_item.assign_status()

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
