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
from simon_arc_lab.task_color_profile import TaskColorProfile
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images
from .work_item import WorkItem
from .work_item_with_previousprediction import WorkItemWithPreviousPrediction
from .work_item_list import WorkItemList
from .work_item_status import WorkItemStatus
from .save_arcprize2024_submission_file import *
from .work_manager_base import WorkManagerBase
from .image_feature import ImageFeature
from .model_gamma1 import ModelGamma1
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

# Solves "1c02dbbe"
FEATURES_6 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.COMPONENT_ALL8,
    ImageFeature.GRAVITY_DRAW_TOP_TO_BOTTOM,
    ImageFeature.GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT,
]

# Makes nice stepwise improvements to the puzzle "3f23242b"
FEATURES_7 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.COMPONENT_ALL8,
    ImageFeature.GRAVITY_DRAW_TOP_TO_BOTTOM,
    ImageFeature.GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT,
    ImageFeature.NUMBER_OF_UNIQUE_COLORS_ALL9,
    ImageFeature.NUMBER_OF_UNIQUE_COLORS_IN_DIAMOND4,
]

FEATURES_8 = [
    ImageFeature.BOUNDING_BOXES,
    ImageFeature.CENTER,
    ImageFeature.SUPPRESS_CENTER_PIXEL_LOOKAROUND,
    ImageFeature.IDENTIFY_OBJECT_SHAPE,
    ImageFeature.COMPONENT_ALL8,
    ImageFeature.GRAVITY_DRAW_TOP_TO_BOTTOM,
    ImageFeature.GRAVITY_DRAW_TOPLEFT_TO_BOTTOMRIGHT,
    # ImageFeature.SHAPE_ALL8,
]

class WorkManagerStepwiseRefinementV4(WorkManagerBase):
    def __init__(self, run_id: str, dataset_id: str, taskset: TaskSet, work_items: list[WorkItemWithPreviousPrediction], incorrect_predictions_jsonl_path: Optional[str] = None):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.taskset = taskset
        self.work_items = work_items
        self.incorrect_predictions_jsonl_path = incorrect_predictions_jsonl_path

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

        # Track incorrect predictions
        incorrect_prediction_metadata = f'run={self.run_id} solver=stepwiserefinement_v4'
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
                    save_dir,
                    track_incorrect_prediction,
                    incorrect_prediction_dataset_id,
                    incorrect_prediction_metadata,
                )

            if work_item.status == WorkItemStatus.CORRECT:
                correct_task_id_set.add(work_item.task.metadata_task_id)
                correct_count = len(correct_task_id_set)
            pbar.set_postfix({'correct': correct_count})

    def process_with_predicted_colorset(
        self, 
        work_item: WorkItemWithPreviousPrediction, 
        profile_index: int, 
        predicted_output_colorset: set[int], 
        predicted_output_color_image: np.array, 
        task_similarity: TaskSimilarity, 
        show: bool, 
        save_dir: Optional[str],
        track_incorrect_prediction: TrackIncorrectPrediction,
        incorrect_prediction_dataset_id: str, 
        incorrect_prediction_metadata: str,
    ):
        # print(f"Processing task: {work_item.task.metadata_task_id} test: {work_item.test_index} profile: {profile_index}")
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
            problem_image = None
            predicted_correctness = None
            predicted_output = None

            if False:
                predicted_output = work_item.previous_predicted_output_image.copy()

            if False:
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
                prediction = ModelGamma1.predict_output(
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
                    predictionj = ModelGamma1.predict_output(
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

            if True:
                last_predicted_correctness = np.zeros_like(work_item.previous_predicted_output_image, dtype=np.uint8)
                noise_level = 90
                the_refinement_index = 1
                try:
                    prediction = ModelGamma1.predict_output(
                        work_item.task, 
                        work_item.test_index, 
                        work_item.previous_predicted_output_image,
                        last_predicted_correctness,
                        the_refinement_index,
                        noise_level,
                        FEATURES_8,
                    )
                except Exception as e:
                    print(f'Error: {e} with task {work_item.task.metadata_task_id} test: {work_item.test_index} uniqueid: {work_item.unique_id}')
                    return
                n_predicted_images = prediction.images(1)
                best_image = n_predicted_images[0]
                confidence_map = prediction.confidence_map()
                entropy_map = prediction.entropy_map()

                predicted_output = best_image

                work_item.predicted_output_image = best_image
                work_item.assign_status()

                if track_incorrect_prediction is not None:
                    track_incorrect_prediction.track_incorrect_prediction_with_workitem(
                        work_item,
                        incorrect_prediction_dataset_id, 
                        predicted_output,
                        incorrect_prediction_metadata
                    )

            # predicted_output = vote_image.copy()
            width = 0
            height = 0
            if predicted_output is not None:
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

            if False:
                validate_result = ModelGamma1.validate_output(
                    work_item.task, 
                    work_item.test_index, 
                    predicted_output,
                    refinement_index, 
                    noise_level,
                    validator_features[refinement_index % len(validator_features)]
                )
                predicted_correctness = validate_result.images(1)[0]
                confidence_map = validate_result.confidence_map()
                entropy_map = validate_result.entropy_map()

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


            if False:
                last_predicted_output = predicted_output
                last_predicted_correctness = predicted_correctness
                score = task_similarity.measure_test_prediction(predicted_output, work_item.test_index)
                # print(f"task: {work_item.task.metadata_task_id} score: {score} refinement_index: {refinement_index} noise_level: {noise_level}")
                image_and_score.append((predicted_output, score))

            if False:
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

            temp_work_item = WorkItemWithPreviousPrediction(
                work_item.task.clone(), 
                work_item.test_index,
                work_item.previous_predicted_output_image.copy(),
                work_item.unique_id 
            )
            temp_work_item.predicted_output_image = predicted_output
            temp_work_item.assign_status()
            if show:
                temp_work_item.show()
            if save_dir is not None:
                temp_work_item.show(save_dir)
            
            title_image_list = []
            title_image_list.append(('arc', 'Input', temp_work_item.task.test_input(temp_work_item.test_index)))
            title_image_list.append(('arc', 'Output', temp_work_item.task.test_output(temp_work_item.test_index)))
            title_image_list.append(('arc', 'Colors', predicted_output_color_image))
            if work_item.previous_predicted_output_image is not None:
                title_image_list.append(('arc', 'Prev Prediction', work_item.previous_predicted_output_image))
            # if best_image is not None:
            #     title_image_list.append(('arc', 'Best', best_image))
            # if vote_image is not None:
            #     title_image_list.append(('arc', 'Vote', vote_image))
            # if second_best_image is not None:
            #     title_image_list.append(('arc', 'Second', second_best_image))
            title_image_list.append(('arc', 'This Prediction', predicted_output))
            if predicted_correctness is not None:
                title_image_list.append(('heatmap', 'Valid', predicted_correctness))
            if problem_image is not None:
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
                f'uniqueid{work_item.unique_id}',
                temp_work_item.status.to_string(),
            ]
            filename_items = [item for item in filename_items_optional if item is not None]
            filename = '_'.join(filename_items) + '.png'

            # Format the title
            title = f'{work_item.task.metadata_task_id} test{work_item.test_index} profile{profile_index} step{refinement_index} uniqueid{work_item.unique_id}' 

            # Save the image to disk or show it.
            if show:
                image_file_path = None
            else:
                image_file_path = os.path.join(save_dir, filename)
            show_multiple_images(title_image_list, title=title, save_path=image_file_path)

        # if len(image_and_score) == 0:
        #     print(f'No predictions for task {work_item.task.metadata_task_id} test: {work_item.test_index}')
        #     return
        # best_image, best_score = max(image_and_score, key=lambda x: x[1])
        # # print(f"task: {work_item.task.metadata_task_id} best_score: {best_score}")

        # work_item.predicted_output_image = best_image
        # work_item.assign_status()

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
