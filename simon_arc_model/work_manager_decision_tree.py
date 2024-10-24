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
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images
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

FEATURES_3 = [
    DecisionTreeFeature.COMPONENT_ALL8,
    DecisionTreeFeature.IMAGE_MASS_COMPARE_ADJACENT_ROWCOL2,
    DecisionTreeFeature.OBJECT_ID_RAY_LIST,
]

class WorkManagerDecisionTree(WorkManagerBase):
    def __init__(self, model: any, taskset: TaskSet, cache_dir: Optional[str] = None):
        self.taskset = taskset
        self.work_items = WorkManagerDecisionTree.create_work_items(taskset)
        self.cache_dir = cache_dir

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
        noise_levels = [100, 0, 0, 0]
        number_of_refinements = len(noise_levels)

        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        for original_work_item in pbar:
            work_item = original_work_item

            ts = TaskSimilarity.create_with_task(work_item.task)

            image_and_score = []

            last_predicted_output = None
            last_predicted_correctness = None
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
                    # print(f"Predicting task: {work_item.task.metadata_task_id} test: {work_item.test_index} refinement: {refinement_index} last_predicted_output: {last_predicted_output is not None} last_predicted_correctness: {last_predicted_correctness is not None}")
                    if last_predicted_output is not None:
                        if last_predicted_correctness is not None:
                            assert last_predicted_output.shape == last_predicted_correctness.shape
                            
                    predicted_output = DecisionTreeUtil.predict_output(
                        work_item.task, 
                        work_item.test_index, 
                        last_predicted_output,
                        last_predicted_correctness,
                        refinement_index, 
                        noise_level,
                        set(FEATURES_2)
                    )
                    if cache_file is not None:
                        np.save(cache_file, predicted_output)

                predicted_correctness = DecisionTreeUtil.validate_output(
                    work_item.task, 
                    work_item.test_index, 
                    predicted_output,
                    refinement_index, 
                    noise_level,
                    set(FEATURES_3)
                )

                last_predicted_output = predicted_output
                last_predicted_correctness = predicted_correctness
                score = ts.measure_test_prediction(predicted_output, work_item.test_index)
                # print(f"task: {work_item.task.metadata_task_id} score: {score} refinement_index: {refinement_index} noise_level: {noise_level}")
                image_and_score.append((predicted_output, score))

                expected_output = work_item.task.test_output(work_item.test_index)
                assert expected_output.shape == predicted_output.shape
                assert expected_output.shape == predicted_correctness.shape
                height, width = predicted_output.shape

                problem_image = np.zeros((height, width), dtype=np.float32)
                for y in range(height):
                    for x in range(width):
                        is_same = predicted_output[y, x] == expected_output[y, x]
                        is_correct = predicted_correctness[y, x] == 1
                        if is_same == False and is_correct == True:
                            # Worst case scenario, the validator was unable to identify this bad pixel.
                            # Thus there is no way for the predictor to ever repair this pixel.
                            value = 0.0
                        else:
                            value = 1.0
                        problem_image[y, x] = value

                temp_work_item = WorkItem(
                    work_item.task.clone(), 
                    work_item.test_index, 
                    refinement_index, 
                    PredictOutputDoNothing()
                )
                temp_work_item.predicted_output_image = predicted_output
                temp_work_item.assign_status()
                # if show:
                #     temp_work_item.show()
                # if save_dir is not None:
                #     temp_work_item.show(save_dir)
                
                title_image_list = []
                title_image_list.append(('arc', 'Input', temp_work_item.task.test_input(temp_work_item.test_index)))
                title_image_list.append(('arc', 'Output', temp_work_item.task.test_output(temp_work_item.test_index)))
                title_image_list.append(('arc', 'Predict', predicted_output))
                title_image_list.append(('heatmap', 'Valid', predicted_correctness))
                title_image_list.append(('heatmap', 'Problem', problem_image))

                # Format the filename for the image, so it contains the task id, test index, and refinement index.
                filename_items_optional = [
                    work_item.task.metadata_task_id,
                    f'test{work_item.test_index}',
                    f'step{refinement_index}',
                    temp_work_item.status.to_string(),
                ]
                filename_items = [item for item in filename_items_optional if item is not None]
                filename = '_'.join(filename_items) + '.png'

                # Format the title
                title = f'{work_item.task.metadata_task_id} test{work_item.test_index} step{refinement_index}'

                # Save the image to disk or show it.
                if show:
                    image_file_path = None
                else:
                    image_file_path = os.path.join(save_dir, filename)
                show_multiple_images(title_image_list, title=title, save_path=image_file_path)

            best_image, best_score = max(image_and_score, key=lambda x: x[1])
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
