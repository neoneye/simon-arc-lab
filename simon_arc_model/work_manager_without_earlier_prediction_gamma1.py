import os
from tqdm import tqdm
import numpy as np
from typing import Optional
from simon_arc_lab.image_distort import *
from simon_arc_lab.image_noise import *
from simon_arc_lab.image_vote import *
from simon_arc_lab.task import Task
from simon_arc_lab.task_mutator import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result
from .work_item import WorkItem
from .work_item_with_refinementstep import WorkItemWithRefinementStep
from .work_item_list import WorkItemList
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *
from .work_manager_base import WorkManagerBase
from .model_gamma1 import ModelGamma1
from .image_feature import ImageFeature
from .track_incorrect_prediction import TrackIncorrectPrediction

# ARC-AGI training=40, evaluation=19
FEATURES_1 = [
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.HISTOGRAM_DIAGONAL,
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.HISTOGRAM_VALUE,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL,
    ImageFeature.BOUNDING_BOXES,
]

# ARC-AGI training=36, evaluation=17
FEATURES_2 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.EROSION_ALL8,
    ImageFeature.HISTOGRAM_ROWCOL,
]

# ARC-AGI training=22, evaluation=3
FEATURES_3 = [
    ImageFeature.COMPONENT_ALL8,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2,
    ImageFeature.OBJECT_ID_RAY_LIST,
]

# ARC-AGI training=34, evaluation=7
FEATURES_4 = [
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_CORNER4, 
    ImageFeature.EROSION_ROWCOL,
]

# ARC-AGI training=36, evaluation=14
FEATURES_5 = [
    ImageFeature.CENTER, 
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_NEAREST4, 
    ImageFeature.HISTOGRAM_ROWCOL,
]

# ARC-AGI training=50, evaluation=19
FEATURES_6 = [
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.HISTOGRAM_DIAGONAL,
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5,
    ImageFeature.HISTOGRAM_VALUE,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2,
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.SHAPE_ALL8,
]

# ARC-AGI training=50, evaluation=19
FEATURES_6_B = [
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.HISTOGRAM_DIAGONAL,
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND5,
    ImageFeature.HISTOGRAM_VALUE,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2,
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.SHAPE_ALL8,
    ImageFeature.LONELY_PIXELS,
]

# ARC-AGI training=39, evaluation=17
FEATURES_7 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.EROSION_ALL8,
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.SHAPE_ALL8,
]

# ARC-AGI training=37, evaluation=6
FEATURES_8 = [
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_CORNER4, 
    ImageFeature.EROSION_ROWCOL,
    ImageFeature.SHAPE_ALL8,
]

# ARC-AGI training=39, evaluation=12
FEATURES_9 = [
    ImageFeature.CENTER, 
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_NEAREST4, 
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.SHAPE_ALL8,
]

# ARC-AGI training=28, evaluation=4, solves many different puzzles
FEATURES_10 = [
    ImageFeature.GRAVITY_MOVE_TOP_TO_BOTTOM,
    ImageFeature.GRAVITY_DRAW_TOP_TO_BOTTOM,
]

# ARC-AGI training=?, evaluation=8
FEATURES_11 = [
    ImageFeature.NUMBER_OF_UNIQUE_COLORS_ALL9,
    ImageFeature.BOUNDING_BOXES,
]

# ARC-AGI training=?, evaluation=12
FEATURES_6 = [
    ImageFeature.LONELY_PIXELS,
    ImageFeature.HISTOGRAM_DIAGONAL,
    ImageFeature.HISTOGRAM_ROWCOL,
]

class WorkManagerWithoutEarlierPredictionGamma1(WorkManagerBase):
    def __init__(self, run_id: str, dataset_id: str, taskset: TaskSet, incorrect_predictions_jsonl_path: Optional[str] = None):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.taskset = taskset
        self.work_items = WorkManagerWithoutEarlierPredictionGamma1.create_work_items(taskset)
        self.incorrect_predictions_jsonl_path = incorrect_predictions_jsonl_path

    @classmethod
    def create_work_items(cls, taskset: TaskSet) -> list['WorkItem']:
        work_items = []
        for task in taskset.tasks:
            if task.has_same_input_output_size_for_all_examples() == False:
                continue

            for test_index in range(task.count_tests):
                work_item = WorkItem(task, test_index)
                work_items.append(work_item)
        return work_items

    def truncate_work_items(self, max_count: int):
        self.work_items = self.work_items[:max_count]

    def discard_items_with_too_long_prompts(self, max_prompt_length: int):
        pass

    def discard_items_with_too_short_prompts(self, min_prompt_length: int):
        pass

    def discard_items_where_predicted_output_is_identical_to_the_input(self):
        self.work_items = WorkItemList.discard_items_where_predicted_output_is_identical_to_the_input(self.work_items)
    
    def process_all_work_items(self, show: bool = False, save_dir: Optional[str] = None):
        if save_dir is not None:
            print(f'Saving images to directory: {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

        features = set(FEATURES_6)

        # Track incorrect predictions
        features_pretty = ImageFeature.names_joined_with_comma(features)
        incorrect_prediction_metadata = f'run={self.run_id} solver=gamma1 features={features_pretty}'
        incorrect_prediction_dataset_id = self.dataset_id
        if self.incorrect_predictions_jsonl_path is not None:
            track_incorrect_prediction = TrackIncorrectPrediction.load_from_jsonl(self.incorrect_predictions_jsonl_path)
        else:
            track_incorrect_prediction = None

        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        for original_work_item in pbar:
            work_item = original_work_item

            last_predicted_output = None
            noise_level = 100
            refinement_index = 0
            previous_prediction_mask = None
            try:
                predicted_output_result = ModelGamma1.predict_output(
                    work_item.task, 
                    work_item.test_index, 
                    last_predicted_output,
                    previous_prediction_mask,
                    refinement_index, 
                    noise_level,
                    features
                )
            except Exception as e:
                # if the error text contains Soft-error, then it is a soft error, and we can skip the task
                if 'Soft-error' in str(e):
                    print(f"Soft-error for task {work_item.task.metadata_task_id} test={work_item.test_index}. Error: {e}")
                    continue
                raise e

            predicted_output = predicted_output_result.images(1)[0]

            temp_work_item = WorkItemWithRefinementStep(
                work_item.task.clone(), 
                work_item.test_index, 
                refinement_index
            )
            temp_work_item.predicted_output_image = predicted_output
            temp_work_item.assign_status()
            if show:
                temp_work_item.show()
            if save_dir is not None:
                temp_work_item.show(save_dir)

            if track_incorrect_prediction is not None:
                track_incorrect_prediction.track_incorrect_prediction_with_workitem(
                    temp_work_item,
                    incorrect_prediction_dataset_id, 
                    predicted_output,
                    incorrect_prediction_metadata
                )

            work_item.predicted_output_image = predicted_output
            work_item.assign_status()

            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
                correct_count = len(correct_task_id_set)
            pbar.set_postfix({'correct': correct_count})

    def summary(self):
        correct_task_id_set = set()
        for work_item in self.work_items:
            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
        correct_count = len(correct_task_id_set)
        print(f'Number of correct solutions: {correct_count}')

    def save_arcprize2024_submission_file(self, path_to_json_file: str):
        json_dict = collect_predictions_as_arcprize2024_submission_dict(self.taskset, self.work_items)
        save_arcprize2024_submission_file(path_to_json_file, json_dict)
