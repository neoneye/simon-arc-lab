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
from .model_beta1 import ModelBeta1, ImageFeature
from .track_incorrect_prediction import TrackIncorrectPrediction

# Correct 59, Solves 1 of the hidden ARC tasks
# ARC-AGI training=41, evaluation=17
FEATURES_1 = [
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.HISTOGRAM_DIAGONAL,
    ImageFeature.HISTOGRAM_ROWCOL,
    ImageFeature.HISTOGRAM_VALUE,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL,
    ImageFeature.BOUNDING_BOXES,
]

# Correct 59, Solves 1 of the hidden ARC tasks
# ARC-AGI training=39, evaluation=20
FEATURES_2 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.COMPONENT_NEAREST4,
    ImageFeature.EROSION_ALL8,
    ImageFeature.HISTOGRAM_ROWCOL,
]

FEATURES_3 = [
    ImageFeature.COMPONENT_ALL8,
    ImageFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2,
    ImageFeature.OBJECT_ID_RAY_LIST,
]

# Correct 47
FEATURES_4 = [
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_CORNER4, 
    ImageFeature.EROSION_ROWCOL,
]

# Correct 48
FEATURES_5 = [
    ImageFeature.CENTER, 
    ImageFeature.COMPONENT_NEAREST4, 
    ImageFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    ImageFeature.EROSION_NEAREST4, 
    ImageFeature.HISTOGRAM_ROWCOL,
]

class WorkManagerDecisionTree(WorkManagerBase):
    def __init__(self, run_id: str, dataset_id: str, taskset: TaskSet, cache_dir: Optional[str] = None, incorrect_predictions_jsonl_path: Optional[str] = None):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.taskset = taskset
        self.work_items = WorkManagerDecisionTree.create_work_items(taskset)
        self.cache_dir = cache_dir
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

        # noise_levels = [95, 90, 85, 80, 75, 70, 65]
        # noise_levels = [95, 90]
        # noise_levels = [100, 95, 90]
        noise_levels = [100]
        number_of_refinements = len(noise_levels)

        features = set(FEATURES_1)

        # Track incorrect predictions
        features_pretty = ImageFeature.names_joined_with_comma(features)
        incorrect_prediction_metadata = f'run={self.run_id} solver=decisiontree_v1 features={features_pretty}'
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

            # ts = TaskSimilarity.create_with_task(work_item.task)

            image_and_score = []

            last_predicted_output = None
            for refinement_index in range(number_of_refinements):
                noise_level = noise_levels[refinement_index]
                # print(f"Refinement {refinement_index+1}/{number_of_refinements} noise_level={noise_level}")
                predicted_output = None
                cache_file = None
                if self.cache_dir is not None:
                    if refinement_index == 0:
                        cache_file = os.path.join(self.cache_dir, f'{work_item.task.metadata_task_id}_{work_item.test_index}.npy')
                        if os.path.isfile(cache_file):
                            predicted_output = np.load(cache_file)
                            # print(f"Loaded from cache: {cache_file}")
                if predicted_output is None:
                    previous_prediction_mask = None
                    try:
                        predicted_output_result = ModelBeta1.predict_output(
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
                            break
                        raise e

                    predicted_output = predicted_output_result.images(1)[0]
                    if cache_file is not None:
                        np.save(cache_file, predicted_output)

                last_predicted_output = predicted_output
                # score = ts.measure_test_prediction(predicted_output, work_item.test_index)
                # print(f"task: {work_item.task.metadata_task_id} score: {score} refinement_index: {refinement_index} noise_level: {noise_level}")
                # image_and_score.append((predicted_output, score))
                best_image = predicted_output

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

            # best_image, best_score = max(image_and_score, key=lambda x: x[1])
            # print(f"task: {work_item.task.metadata_task_id} best_score: {best_score}")

            work_item.predicted_output_image = best_image
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
