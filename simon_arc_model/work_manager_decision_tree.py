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
from simon_arc_lab.task_color_profile import TaskColorProfile
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

# Correct 47
FEATURES_4 = [
    DecisionTreeFeature.COMPONENT_NEAREST4, 
    DecisionTreeFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    DecisionTreeFeature.EROSION_CORNER4, 
    DecisionTreeFeature.EROSION_ROWCOL,
]

# Correct 48
FEATURES_5 = [
    DecisionTreeFeature.CENTER, 
    DecisionTreeFeature.COMPONENT_NEAREST4, 
    DecisionTreeFeature.COUNT_NEIGHBORS_WITH_SAME_COLOR, 
    DecisionTreeFeature.EROSION_NEAREST4, 
    DecisionTreeFeature.HISTOGRAM_ROWCOL,
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

        correct_count = 0
        correct_task_id_set = set()
        pbar = tqdm(self.work_items, desc="Processing work items")
        for original_work_item in pbar:
            work_item = original_work_item

            task_similarity = TaskSimilarity.create_with_task(work_item.task)

            profile = TaskColorProfile(work_item.task)
            task_color_profile_prediction = profile.predict_output_colors_for_test_index(work_item.test_index)
            predicted_output_color_image = task_color_profile_prediction.to_image()

            for colorprofile_index, (certain, colorset) in enumerate(task_color_profile_prediction.certain_colorset_list):
                self.process_with_predicted_colorset(
                    work_item,
                    colorprofile_index,
                    colorset, 
                    predicted_output_color_image, 
                    task_similarity, 
                    show, 
                    save_dir
                )

            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
                correct_count = len(correct_task_id_set)
            pbar.set_postfix({'correct': correct_count})

    def process_with_predicted_colorset(self, work_item: WorkItem, profile_index: int, predicted_output_colorset: set[int], predicted_output_color_image: np.array, task_similarity: TaskSimilarity, show: bool, save_dir: Optional[str]):
        # noise_levels = [95, 90, 85, 80, 75, 70, 65]
        # noise_levels = [95, 90]
        # noise_levels = [100, 95, 90]
        noise_levels = [100]
        # noise_levels = [100, 0]
        # noise_levels = [100, 0, 0]
        number_of_refinements = len(noise_levels)

        predict_output_features = [
            set(FEATURES_3),
            set(FEATURES_4),
            set(FEATURES_5),
            set(FEATURES_1),
            set(FEATURES_2),
        ]

        validator_features = [
            set(FEATURES_4),
            set(FEATURES_5),
            set(FEATURES_2),
            set(FEATURES_1),
            set(FEATURES_3),
        ]


        image_and_score = []

        last_predicted_output = None
        last_predicted_correctness = None
        for refinement_index in range(number_of_refinements):
            noise_level = noise_levels[refinement_index]
            # print(f"Refinement {refinement_index+1}/{number_of_refinements} noise_level={noise_level}")
            # print(f"Predicting task: {work_item.task.metadata_task_id} test: {work_item.test_index} refinement: {refinement_index} last_predicted_output: {last_predicted_output is not None} last_predicted_correctness: {last_predicted_correctness is not None}")
            if last_predicted_output is not None:
                if last_predicted_correctness is not None:
                    assert last_predicted_output.shape == last_predicted_correctness.shape

            vote_image = None
            best_image = None
            second_best_image = None
            third_best_image = None
            fourth_best_image = None
            confidence_map = None
            entropy_map = None

            if True:
                expected_output = work_item.task.test_output(work_item.test_index)
                predicted_output = expected_output.copy()
                height, width = expected_output.shape
                xy_positions = []
                for y in range(height):
                    for x in range(width):
                        xy_positions.append((x, y))
                random.Random(refinement_index).shuffle(xy_positions)
                x, y = xy_positions[0]
                color = expected_output[y, x]
                candidate_color_set = set(predicted_output_colorset)
                candidate_color_set.discard(color)
                if len(candidate_color_set) == 0:
                    print(f'No candidate colors for task {work_item.task.metadata_task_id} test: {work_item.test_index} at position {x}, {y}')
                    return
                color = random.Random(y * width + x).choice(list(candidate_color_set))
                predicted_output[y, x] = color

            if False:
                prediction = DecisionTreeUtil.predict_output(
                    work_item.task, 
                    work_item.test_index, 
                    last_predicted_output,
                    last_predicted_correctness,
                    refinement_index, 
                    noise_level,
                    predict_output_features[refinement_index % len(predict_output_features)]
                )
                n_predicted_images = prediction.images(4)
                best_image = n_predicted_images[0]
                second_best_image = n_predicted_images[1]
                third_best_image = n_predicted_images[2]
                fourth_best_image = n_predicted_images[3]
                confidence_map = prediction.confidence_map()
                entropy_map = prediction.entropy_map()

                best_images = [best_image]
                for j in range(4):
                    predictionj = DecisionTreeUtil.predict_output(
                        work_item.task, 
                        work_item.test_index, 
                        last_predicted_output,
                        last_predicted_correctness,
                        refinement_index, 
                        noise_level,
                        predict_output_features[(refinement_index + j + 1) % len(predict_output_features)]
                    )
                    n_predicted_imagesj = predictionj.images(1)
                    best_imagej = n_predicted_imagesj[0]
                    best_images.append(best_imagej)

                vote_image = image_vote(best_images)

            # predicted_output = vote_image.copy()
            height, width = predicted_output.shape
            if False:
                count_repair = 0
                for y in range(height):
                    for x in range(width):
                        color = predicted_output[y, x]
                        if color not in predicted_output_colorset:
                            color = random.Random(y * width + x).choice(list(predicted_output_colorset))
                            predicted_output[y, x] = color
                            count_repair += 1
                print(f'task {work_item.task.metadata_task_id} test: {work_item.test_index} repaired {count_repair} pixels based on predicted output colorset')

            predicted_correctness = DecisionTreeUtil.validate_output(
                work_item.task, 
                work_item.test_index, 
                predicted_output,
                refinement_index, 
                noise_level,
                validator_features[refinement_index % len(validator_features)]
            )

            expected_output = work_item.task.test_output(work_item.test_index)
            assert expected_output.shape == predicted_output.shape
            assert expected_output.shape == predicted_correctness.shape

            if False:
                count_repair = 0
                for y in range(height):
                    for x in range(width):
                        if predicted_correctness[y, x] == 0:
                            predicted_output[y, x] = second_best_image[y, x]
                            count_repair += 1
                # print(f'repaired {count_repair} pixels')

            if False:
                count_repair = 0
                for y in range(height):
                    for x in range(width):
                        if predicted_correctness[y, x] > 0:
                            continue
                        color = predicted_output[y, x]
                        # copy the predicted_output_colorset, and remove the current color from the set
                        candidate_color_set = set(predicted_output_colorset)
                        candidate_color_set.discard(color)
                        if len(candidate_color_set) == 0:
                            continue
                        color = random.Random(y * width + x).choice(list(candidate_color_set))
                        predicted_output[y, x] = color
                        count_repair += 1
                print(f'task {work_item.task.metadata_task_id} test: {work_item.test_index} repaired {count_repair} pixels based on validation')

            if False:
                count_correct1 = 0
                count_correct2 = 0
                count_correct3 = 0
                count_correct4 = 0
                count_incorrect = 0
                for y in range(height):
                    for x in range(width):
                        color = expected_output[y, x]
                        if best_image[y, x] == color:
                            count_correct1 += 1
                        elif second_best_image[y, x] == color:
                            count_correct2 += 1
                        elif third_best_image[y, x] == color:
                            count_correct3 += 1
                        elif fourth_best_image[y, x] == color:
                            count_correct4 += 1
                        else:
                            count_incorrect += 1
                # print(f'correct {count_correct} incorrect {count_incorrect}')
            if False:
                correct_list = [count_correct1, count_correct2, count_correct3, count_correct4]
                if count_incorrect == 0:
                    if count_correct4 > 0:
                        rank = 4
                    if count_correct3 > 0:
                        rank = 3
                    elif count_correct2 > 0:
                        rank = 2
                    elif count_correct1 > 0:
                        rank = 1
                    else:
                        rank = None
                    print(f'good task: {work_item.task.metadata_task_id} test: {work_item.test_index} correct_list: {correct_list} rank: {rank}')
                else:
                    count_correct = count_correct1 + count_correct2 + count_correct3 + count_correct4
                    percent = count_correct * 100 // (count_correct + count_incorrect)
                    print(f'bad task: {work_item.task.metadata_task_id} test: {work_item.test_index} count_incorrect: {count_incorrect} correct_list: {correct_list} correctness_percentage: {percent}')


            last_predicted_output = predicted_output
            last_predicted_correctness = predicted_correctness
            score = task_similarity.measure_test_prediction(predicted_output, work_item.test_index)
            # print(f"task: {work_item.task.metadata_task_id} score: {score} refinement_index: {refinement_index} noise_level: {noise_level}")
            image_and_score.append((predicted_output, score))

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
            title_image_list.append(('arc', 'Colors', predicted_output_color_image))
            if best_image is not None:
                title_image_list.append(('arc', 'Best', best_image))
            if vote_image is not None:
                title_image_list.append(('arc', 'Vote', vote_image))
            # if second_best_image is not None:
            #     title_image_list.append(('arc', 'Second', second_best_image))
            title_image_list.append(('arc', 'Predict', predicted_output))
            title_image_list.append(('heatmap', 'Valid', predicted_correctness))
            title_image_list.append(('heatmap', 'Problem', problem_image))
            if confidence_map is not None:
                title_image_list.append(('heatmap', 'Confidence', confidence_map))
            if entropy_map is not None:
                title_image_list.append(('heatmap', 'Entropy', entropy_map))

            # Format the filename for the image, so it contains the task id, test index, and refinement index.
            filename_items_optional = [
                work_item.task.metadata_task_id,
                f'test{work_item.test_index}',
                f'profile{profile_index}',
                f'step{refinement_index}',
                temp_work_item.status.to_string(),
            ]
            filename_items = [item for item in filename_items_optional if item is not None]
            filename = '_'.join(filename_items) + '.png'

            # Format the title
            title = f'{work_item.task.metadata_task_id} test{work_item.test_index} profile{profile_index} step{refinement_index}'

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
