from datetime import datetime
import os
import sys
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
import matplotlib.pyplot as plt
from math import sqrt
from enum import Enum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images
from simon_arc_lab.image_noise import *

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

class DataPoint(Enum):
    PAIR_ID = 0
    PIXEL_VALUE = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5
    PIXEL_VALUES = 6

def datapoints_from_image(pair_id: int, image: np.array) -> list:
    height, width = image.shape
    data = []
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            values = [
                pair_id,
                pixel_value,
                x,
                y,
                width,
                height,
            ]
            for dx in range(3):
                for dy in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    x2 = x + dx
                    y2 = y + dy
                    pixel_value2 = 10
                    if x2 >= 0 and x2 < width and y2 >= 0 and y2 < height:
                        pixel_value2 = image[y2, x2]
                    values.append(pixel_value2)
            data.append(values)
    return data

def sample_data(input_data: list, target_data: list, rng) -> list:
    """
    The input_data and target_data can have different lengths. 
    This function makes sure the resulting list have the same length.
    Sample N items from both lists, until all items have been processed.
    """
    # Sample max N times per item.
    input_data_sample_count = np.zeros(len(input_data), dtype=int)
    target_data_sample_count = np.zeros(len(target_data), dtype=int)

    # The unvisited indexes.
    input_data_indexes = np.arange(len(input_data))
    target_data_indexes = np.arange(len(target_data))

    number_of_values_per_sample = 10
    number_of_samples = 400

    input_target_pairs = []
    for i in range(number_of_samples):
        if len(input_data_indexes) < number_of_values_per_sample:
            break

        input_data_sample_indexes = rng.choice(input_data_indexes, number_of_values_per_sample)
        for index in input_data_sample_indexes:
            input_data_sample_count[index] += 1
            if input_data_sample_count[index] == number_of_values_per_sample:
                input_data_indexes = np.delete(input_data_indexes, np.where(input_data_indexes == index))

        # print(f"input_data_sample_indexes: {input_data_sample_indexes}")
        input_data_samples = [input_data[index] for index in input_data_sample_indexes]
        # print(f"input_data_samples: {input_data_samples}")

        if len(target_data_indexes) < number_of_values_per_sample:
            break

        target_data_sample_indexes = rng.choice(target_data_indexes, number_of_values_per_sample)
        for index in target_data_sample_indexes:
            target_data_sample_count[index] += 1
            if target_data_sample_count[index] == number_of_values_per_sample:
                target_data_indexes = np.delete(target_data_indexes, np.where(target_data_indexes == index))
        
        # print(f"target_data_sample_indexes: {target_data_sample_indexes}")
        target_data_samples = [target_data[index] for index in target_data_sample_indexes]
        # print(f"target_data_samples: {target_data_samples}")

        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        input_target_pairs.append((input_data_samples, target_data_samples))
    return input_target_pairs

def count_correct_with_pairs(input_target_pairs: list) -> tuple[int, int]:
    count_correct = 0
    count_total = 0
    for input_data_samples, target_data_samples in input_target_pairs:
        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        n = len(input_data_samples)
        # print(f"n: {n}")
        this_count_correct = 0
        for y in range(n):
            is_target_correct = False
            for x in range(n):
                input_values = input_data_samples[y]
                target_values = target_data_samples[x]

                input_value = input_values[1]
                target_value = target_values[1]

                is_correct = input_value == target_value

                if is_correct:
                    is_target_correct = True
            if is_target_correct:
                this_count_correct += 1
        
        count_correct += (this_count_correct / n)
        count_total += 1
    
    if count_total == 0:
        raise ValueError(f"count_total is zero")

    return count_correct, count_total

def xs_ys_from_input_target_pairs(input_target_pairs: list) -> tuple[list, list]:
    xs = []
    ys = []
    extra = []
    for input_data_samples, target_data_samples in input_target_pairs:
        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        n = len(input_data_samples)
        # print(f"n: {n}")
        for y in range(n):
            for x in range(n):
                input_values = input_data_samples[y]
                target_values = target_data_samples[x]

                input_pair_index = input_values[DataPoint.PAIR_ID.value]
                input_value = input_values[DataPoint.PIXEL_VALUE.value]
                input_x = input_values[DataPoint.X.value]
                input_y = input_values[DataPoint.Y.value]
                input_width = input_values[DataPoint.WIDTH.value]
                input_height = input_values[DataPoint.HEIGHT.value]
                input_pixel_values2 = input_values[DataPoint.PIXEL_VALUES.value:-1]
                input_x_rev = input_width - input_x - 1
                input_y_rev = input_height - input_y - 1

                target_pair_index = target_values[DataPoint.PAIR_ID.value]
                target_value = target_values[DataPoint.PIXEL_VALUE.value]
                target_x = target_values[DataPoint.X.value]
                target_y = target_values[DataPoint.Y.value]
                target_width = target_values[DataPoint.WIDTH.value]
                target_height = target_values[DataPoint.HEIGHT.value]
                target_x_rev = target_width - target_x - 1
                target_y_rev = target_height - target_y - 1

                is_correct = input_value == target_value

                dx = input_x - target_x
                dy = input_y - target_y
                distance0 = abs(dx) + abs(dy)
                distance1 = sqrt(dx * dx + dy * dy)

                same_pair_id = 1 if input_pair_index == target_pair_index else 0

                # one hot encoding of input_value
                one_hot_input_value = np.zeros(10, dtype=int)
                one_hot_input_value[input_value] = 1
                one_hot_input_value = one_hot_input_value.tolist()

                # one hot encoding of target_value
                one_hot_target_value = np.zeros(10, dtype=int)
                one_hot_target_value[target_value] = 1
                one_hot_target_value = one_hot_target_value.tolist()

                xs_item = [
                    target_x,
                    target_y,
                    input_pair_index,
                    target_pair_index,
                    same_pair_id,
                    input_value,
                    input_x,
                    input_y,
                    input_width,
                    input_height,
                    # target_value,
                    target_width,
                    target_height,
                    dx,
                    dy,
                    input_x_rev,
                    input_y_rev,
                    target_x_rev,
                    target_y_rev,
                    # distance0,
                    distance1,
                ]
                xs_item += one_hot_input_value
                # xs_item += one_hot_target_value
                xs_item += input_pixel_values2

                ys_item = target_value

                extra_item = [
                    input_pair_index,
                    target_pair_index,
                ]

                xs.append(xs_item)
                ys.append(ys_item)
                extra.append(extra_item)
    return xs, ys, extra

def process_task(task: Task, weights: np.array, save_dir: str):
    # print(f"Processing task '{task.metadata_task_id}'")
    rng = np.random.default_rng(seed=42)

    input_data = []
    for i in range(task.count_examples + task.count_tests):
        image = task.input_images[i]
        input_data += datapoints_from_image(i, image)

    target_data_only_examples = []
    for i in range(task.count_examples):
        image = task.output_images[i]
        target_data_only_examples += datapoints_from_image(i, image)

    output_image_to_verify = task.test_output(0)
    if False:
        output_image_to_verify = image_noise_one_pixel(output_image_to_verify, 42)
    target_data_with_one_test = []
    if True:
        target_data_with_one_test += target_data_only_examples
        target_data_with_one_test += datapoints_from_image(task.count_examples, output_image_to_verify)

    random.Random(0).shuffle(input_data)
    random.Random(1).shuffle(target_data_only_examples)
    # print(f"input_data: {len(input_data)} target_data: {len(target_data)}")

    input_target_pairs = sample_data(input_data, target_data_only_examples, rng)

    random.Random(2).shuffle(input_data)
    random.Random(3).shuffle(target_data_with_one_test)
    input_target_pairs_one_test = sample_data(input_data, target_data_with_one_test, rng)

    random.Random(4).shuffle(input_target_pairs)
    random.Random(5).shuffle(input_target_pairs_one_test)

    count_correct, count_total = count_correct_with_pairs(input_target_pairs)
    if count_total == 0:
        raise ValueError(f"count_total is zero")
    average = count_correct / count_total
    # print(f"average: {average}")
    # print(f"count_correct: {count_correct} of {n}")

    xs, ys, extra = xs_ys_from_input_target_pairs(input_target_pairs)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(xs, ys)

    xs2, ys2, extra2 = xs_ys_from_input_target_pairs(input_target_pairs_one_test)
    predicted_values = clf.predict(xs2)

    if len(predicted_values) != len(ys2):
        raise ValueError(f"predicted_values and ys2 have different lengths. predicted_values len: {len(predicted_values)} ys2 len: {len(ys2)}")

    # print(f"predicted_values: {len(predicted_values)}")
    # print(f"ys2: {len(ys2)}")

    pred_count_correct = 0
    pred_count_incorrect = 0
    for i in range(len(predicted_values)):
        if predicted_values[i] == ys2[i]:
            pred_count_correct += 1
        else:
            pred_count_incorrect += 1
            # print(f"predicted_values[{i}]: {predicted_values[i]} ys2[{i}]: {ys2[i]}")

    # When the classifier checks the unmodified input, it should always be correct.
    # However I'm not at that point yet, there are still many incorrect predictions.
    print(f"task: {task.metadata_task_id} correct: {pred_count_correct} incorrect: {pred_count_incorrect}")

    pred_is_correct = pred_count_incorrect == 0

    predicted_image = None
    if True:
        expected_output_image = task.test_output(0)
        # image = np.zeros_like(expected_output_image, dtype=np.float32)
        color_count_image = []
        for i in range(10):
            image = np.zeros_like(expected_output_image, dtype=np.uint32)
            color_count_image.append(image)
        for i in range(len(predicted_values)):
            xs2_item = xs2[i]
            target_pair_id = extra2[i][1]
            if target_pair_id != task.count_examples:
                continue
            target_x = xs2_item[0]
            target_y = xs2_item[1]
            # v = image[target_y, target_x]
            # if predicted_values[i] == expected_output_image[target_y, target_x]:
            #     v += 1.0
            # image[target_y, target_x] = v
            color = predicted_values[i]
            color_count_image[color][target_y, target_x] += 1
        
        height, width = expected_output_image.shape
        image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                max_color = 10
                max_count = 0
                for i in range(10):
                    count = color_count_image[i][y, x]
                    if count > max_count:
                        max_color = i
                        max_count = count
                image[y, x] = max_color
        # min_value = np.min(image)
        # max_value = np.max(image)
        # diff = max_value - min_value
        # image2 = np.zeros_like(image, dtype=np.float32)
        # if diff > 0.01:
        #     for y in range(image.shape[0]):
        #         for x in range(image.shape[1]):
        #             v = image[y, x]
        #             image2[y, x] = (v - min_value) / diff
        # predicted_image = image2
        predicted_image = image

    # Save the image to disk or show it.
    if True:
        test_pair_index = 0
        title = f"Task {task.metadata_task_id} pair {test_pair_index} average: {average:.2f} correct: {pred_count_correct} incorrect: {pred_count_incorrect}"
        input_image = task.test_input(test_pair_index)
        output_image = output_image_to_verify
        title_image_list = [
            ('arc', 'input', input_image),
            ('arc', 'output', output_image),
            ('arc', 'pred', predicted_image),
        ]
        suffix = 'correct' if pred_is_correct else 'incorrect'
        filename = f'{task.metadata_task_id}_pair{test_pair_index}_{suffix}.png'
        image_file_path = os.path.join(save_dir, filename)
        show_multiple_images(title_image_list, title=title, save_path=image_file_path)

    return (average, pred_is_correct)


weights_width = 100
weights_height = 100
weights = np.random.rand(weights_height, weights_width)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.remove_tasks_by_id(set(['1_3_5_l6aejqqqc1b47pjr5g4']), True)


    # put the average in k bins
    bins = 10
    bin_width = 1 / bins
    bin_values = np.zeros(bins, dtype=float)

    count_pred_is_correct = 0
    for task_index, task in enumerate(taskset.tasks):
        try:
            average, pred_is_correct = process_task(task, weights, save_dir)
            # pass
        except Exception as e:
            print(f"Error processing task {task.metadata_task_id}: {e}")
            continue
        bin_index = int(average / bin_width)
        if bin_index >= bins:
            bin_index = bins - 1
        bin_values[bin_index] += 1

        if pred_is_correct:
            count_pred_is_correct += 1
        if task_index > 100:
            break

    print(f"bin_values: {bin_values}")
    print(f"count_pred_is_correct: {count_pred_is_correct}")

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_run(save_dir, title=gallery_title)
