"""
This model is the most unusual model I have created so far.

- Most of the predicted outputs correspond to one of the example outputs.
- Nearly always the predicted output corresponds to the output from the example pair index with highest index.
- In rare cases, the predicted output is identical to one of the other example outputs.
- In very rare cases, the predicted output is not identical to any of the example outputs, and it actually attempts to make a prediction about the test output.

IDEA: Do a cross over between random sampling and the ModelGamma1 decision tree model. That way it will hopefully yield
better results than just picking one of the example outputs. 

IDEA: Change the `xs_ys_from_input_target_pairs` from returning lists, to instead return a dict.
This dict can be put into a pandas DataFrame, and then the crossover between ModelGamma1 and random sampling can easier be done.
"""
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

# -----------------------------
# Project Setup and Imports
# -----------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images
from simon_arc_lab.image_noise import *
from simon_arc_lab.image_shape3x3_center import *
from simon_arc_lab.image_shape3x3_opposite import *
from simon_arc_lab.pixel_connectivity import *
from simon_arc_lab.connected_component import *
from simon_arc_lab.image_object_mass import *
from simon_arc_lab.image_raytrace_probecolor import *
from simon_arc_lab.histogram import *

# -----------------------------
# Configuration
# -----------------------------

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # Uncomment and add other datasets as needed
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

# -----------------------------
# Enum for DataPoint Indices
# -----------------------------

class DataPoint(Enum):
    PAIR_ID = 0
    PIXEL_VALUE = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5
    PIXEL_VALUES = 6  # Starting index for neighboring pixel values

# -----------------------------
# Feature Extraction Functions
# -----------------------------

def datapoints_from_image(pair_id: int, image: np.array) -> list:
    """
    Extracts features from the input image for each pixel.
    """
    height, width = image.shape
    data = []

    # Define neighbor positions (3x3 grid excluding center)
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]

            # Extract neighboring pixel values
            neighbors = []
            for dy, dx in neighbor_offsets:
                x2 = x + dx
                y2 = y + dy
                if 0 <= x2 < width and 0 <= y2 < height:
                    neighbors.append(image[y2, x2])
                else:
                    neighbors.append(10)  # Using 10 as a default out-of-bounds value

            # Compile feature vector
            values = [
                pair_id,         # PAIR_ID
                pixel_value,     # PIXEL_VALUE
                x,               # X
                y,               # Y
                width,           # WIDTH
                height,          # HEIGHT
            ]
            values.extend(neighbors)  # PIXEL_VALUES (neighbors)

            data.append(values)
    return data

def sample_data(input_data: list, target_data: list, rng) -> list:
    """
    Samples input and target data to create pairs.
    Ensures balanced sampling.
    """
    input_data_sample_count = np.zeros(len(input_data), dtype=int)
    target_data_sample_count = np.zeros(len(target_data), dtype=int)

    input_data_indexes = np.arange(len(input_data))
    target_data_indexes = np.arange(len(target_data))

    number_of_values_per_sample = 10
    number_of_samples = 400

    input_target_pairs = []
    for _ in range(number_of_samples):
        if len(input_data_indexes) < number_of_values_per_sample or len(target_data_indexes) < number_of_values_per_sample:
            break

        input_sample_idxs = rng.choice(input_data_indexes, number_of_values_per_sample, replace=False)
        target_sample_idxs = rng.choice(target_data_indexes, number_of_values_per_sample, replace=False)

        input_samples = [input_data[idx] for idx in input_sample_idxs]
        target_samples = [target_data[idx] for idx in target_sample_idxs]

        # Update sample counts and remove exhausted indices
        for idx in input_sample_idxs:
            input_data_sample_count[idx] += 1
            if input_data_sample_count[idx] == number_of_values_per_sample:
                input_data_indexes = np.delete(input_data_indexes, np.where(input_data_indexes == idx))

        for idx in target_sample_idxs:
            target_data_sample_count[idx] += 1
            if target_data_sample_count[idx] == number_of_values_per_sample:
                target_data_indexes = np.delete(target_data_indexes, np.where(target_data_indexes == idx))

        input_target_pairs.append((input_samples, target_samples))
    return input_target_pairs

def count_correct_with_pairs(input_target_pairs: list) -> tuple[int, int]:
    """
    Counts the number of correct predictions based on input-target pairs.
    """
    count_correct = 0
    count_total = 0
    for input_samples, target_samples in input_target_pairs:
        if len(input_samples) != len(target_samples):
            raise ValueError("Input and target samples must have the same length.")

        n = len(input_samples)
        this_count_correct = 0
        for y in range(n):
            is_target_correct = False
            for x in range(n):
                input_value = input_samples[y][DataPoint.PIXEL_VALUE.value]
                target_value = target_samples[x][DataPoint.PIXEL_VALUE.value]
                if input_value == target_value:
                    is_target_correct = True
                    break
            if is_target_correct:
                this_count_correct += 1

        count_correct += (this_count_correct / n)
        count_total += 1

    if count_total == 0:
        raise ValueError("No samples to evaluate.")

    return count_correct, count_total

def xs_ys_from_input_target_pairs(input_target_pairs: list) -> tuple[list, list, list, list]:
    """
    Converts input-target pairs into feature vectors (X) and labels (y).
    Also returns feature names for later use.
    """
    xs = []
    ys = []
    extra = []
    feature_names = []

    # Define neighbor positions for feature naming
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]

    for input_samples, target_samples in input_target_pairs:
        if len(input_samples) != len(target_samples):
            raise ValueError("Input and target samples must have the same length.")

        for input_sample in input_samples:
            for target_sample in target_samples:
                # Extract features from input and target samples
                input_pair_id = input_sample[DataPoint.PAIR_ID.value]
                input_pixel_value = input_sample[DataPoint.PIXEL_VALUE.value]
                input_x = input_sample[DataPoint.X.value]
                input_y = input_sample[DataPoint.Y.value]
                input_width = input_sample[DataPoint.WIDTH.value]
                input_height = input_sample[DataPoint.HEIGHT.value]
                input_pixel_neighbors = input_sample[DataPoint.PIXEL_VALUES.value:]

                target_pair_id = target_sample[DataPoint.PAIR_ID.value]
                target_pixel_value = target_sample[DataPoint.PIXEL_VALUE.value]
                target_x = target_sample[DataPoint.X.value]
                target_y = target_sample[DataPoint.Y.value]
                target_width = target_sample[DataPoint.WIDTH.value]
                target_height = target_sample[DataPoint.HEIGHT.value]

                # Calculate differences
                dx = input_x - target_x
                dy = input_y - target_y
                distance = sqrt(dx * dx + dy * dy)

                # Determine if pair IDs match
                same_pair_id = int(input_pair_id == target_pair_id)

                # One-Hot Encode input pixel value
                one_hot_input = [0] * 10
                if 0 <= input_pixel_value < 10:
                    one_hot_input[input_pixel_value] = 1
                else:
                    one_hot_input[0] = 1  # Default encoding for out-of-range values

                # Compile feature vector
                features = [
                    target_x,
                    target_y,
                    input_pair_id,
                    target_pair_id,
                    same_pair_id,
                    input_pixel_value,
                    input_x,
                    input_y,
                    input_width,
                    input_height,
                    target_width,
                    target_height,
                    dx,
                    dy,
                    distance
                ]
                features.extend(one_hot_input)
                features.extend(input_pixel_neighbors)

                xs.append(features)
                ys.append(target_pixel_value)

                # Optional: Collect extra information for visualization or debugging
                extra.append([
                    input_pair_id,
                    target_pair_id,
                    target_x,
                    target_y
                ])

    # Define feature names
    base_feature_names = [
        'target_x', 'target_y', 'input_pair_id', 'target_pair_id', 'same_pair_id',
        'input_pixel_value', 'input_x', 'input_y', 'input_width', 'input_height',
        'target_width', 'target_height', 'dx', 'dy', 'distance'
    ]
    one_hot_feature_names = [f'one_hot_input_pixel_{i}' for i in range(10)]
    neighbor_feature_names = [f'pixel_neighbor_{dx}_{dy}' for dy, dx in neighbor_offsets]

    feature_names = base_feature_names + one_hot_feature_names + neighbor_feature_names

    return xs, ys, extra, feature_names

# -----------------------------
# Task Processing Function
# -----------------------------

def process_task(task: Task, weights: np.array, save_dir: str, k_features: int):
    """
    Processes a single ARC task:
    - Extracts features from input and target images.
    - Performs feature selection.
    - Trains a DecisionTreeClassifier.
    - Evaluates and visualizes predictions.
    """
    print(f"Processing task '{task.metadata_task_id}'")
    rng = np.random.default_rng(seed=42)

    # Extract features from input images
    input_data = []
    for i in range(task.count_examples + task.count_tests):
        image = task.input_images[i]
        input_data += datapoints_from_image(i, image)

    # Extract features from target images (only examples)
    target_data_only_examples = []
    for i in range(task.count_examples):
        image = task.output_images[i]
        target_data_only_examples += datapoints_from_image(i, image)

    # Optionally, include test output for verification
    output_image_to_verify = task.test_output(0)
    target_data_with_one_test = target_data_only_examples + datapoints_from_image(task.count_examples, output_image_to_verify)

    # Shuffle data to ensure randomness
    random.Random(0).shuffle(input_data)
    random.Random(1).shuffle(target_data_only_examples)

    # Sample input-target pairs for training
    input_target_pairs = sample_data(input_data, target_data_only_examples, rng)

    # Shuffle and sample input-target pairs for testing
    random.Random(2).shuffle(input_data)
    random.Random(3).shuffle(target_data_with_one_test)
    input_target_pairs_one_test = sample_data(input_data, target_data_with_one_test, rng)

    # Final shuffle to ensure randomness
    random.Random(4).shuffle(input_target_pairs)
    random.Random(5).shuffle(input_target_pairs_one_test)

    # Evaluate correctness of sampled pairs
    count_correct, count_total = count_correct_with_pairs(input_target_pairs)
    average = count_correct / count_total
    print(f"Sampling Correctness: {average:.2f} ({count_correct} out of {count_total})")

    # Prepare training data
    xs, ys, extra, feature_names = xs_ys_from_input_target_pairs(input_target_pairs)
    X = np.array(xs)
    y = np.array(ys)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define feature indices based on feature names
    # Assuming the first 15 features are numerical, next 10 are one-hot, and the last 8 are pixel neighbors
    numerical_indices = list(range(15))
    one_hot_indices = list(range(15, 25))
    pixel_neighbor_indices = list(range(25, 33))

    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_indices),       # Numerical features
            ('onehot', 'passthrough', one_hot_indices),      # One-Hot Encoded features
            ('neighbors', 'passthrough', pixel_neighbor_indices)  # Neighboring pixel values
        ]
    )

    # Define the full pipeline with feature selection and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=mutual_info_classif, k=k_features)),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=15))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate on the validation set
    val_accuracy = pipeline.score(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Prepare test data
    xs2, ys2, extra2, _ = xs_ys_from_input_target_pairs(input_target_pairs_one_test)
    X_test = np.array(xs2)
    y_test = np.array(ys2)

    # Predict on test data
    predicted_values = pipeline.predict(X_test)

    if len(predicted_values) != len(y_test):
        raise ValueError("Mismatch in prediction lengths.")

    # Calculate prediction accuracy
    pred_count_correct = np.sum(predicted_values == y_test)
    pred_count_incorrect = len(y_test) - pred_count_correct
    print(f"Task '{task.metadata_task_id}' Prediction: {pred_count_correct} correct, {pred_count_incorrect} incorrect")

    pred_is_correct = pred_count_incorrect == 0

    # Generate predicted image based on predictions
    expected_output_image = task.test_output(0)
    height, width = expected_output_image.shape
    predicted_image = np.zeros((height, width), dtype=np.uint8)

    # Aggregate predictions for each pixel
    for i in range(len(predicted_values)):
        target_pair_id = extra2[i][1]
        if target_pair_id != task.count_examples:
            continue  # Only process test outputs
        target_x = extra2[i][2]
        target_y = extra2[i][3]
        color = predicted_values[i]
        predicted_image[target_y, target_x] = color

    # Save or display the predicted image
    test_pair_index = 0
    title = f"Task {task.metadata_task_id} | Avg Correct: {average:.2f} | Correct: {pred_count_correct} | Incorrect: {pred_count_incorrect}"
    input_image = task.test_input(test_pair_index)
    output_image = expected_output_image
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

# -----------------------------
# Main Execution Loop
# -----------------------------

# Initialize random weights if needed (not used in current implementation)
weights_width = 100
weights_height = 100
weights = np.random.rand(weights_height, weights_width)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    # Remove specific tasks if needed
    taskset.remove_tasks_by_id(set(['1_3_5_l6aejqqqc1b47pjr5g4']), True)

    # Initialize bins for averaging
    bins = 10
    bin_width = 1 / bins
    bin_values = np.zeros(bins, dtype=int)

    count_pred_is_correct = 0
    max_tasks = 100  # Limit to first 100 tasks for efficiency

    for task_index, task in enumerate(taskset.tasks):
        if task_index >= max_tasks:
            break
        try:
            average, pred_is_correct = process_task(task, weights, save_dir, k_features=20)
            # pass
        except Exception as e:
            print(f"Error processing task {task.metadata_task_id}: {e}")
            continue

        # Assign to bins based on average correctness
        bin_index = int(average / bin_width)
        if bin_index >= bins:
            bin_index = bins - 1
        bin_values[bin_index] += 1

        if pred_is_correct:
            count_pred_is_correct += 1

    print(f"bin_values: {bin_values}")
    print(f"count_pred_is_correct: {count_pred_is_correct}")

    gallery_title = f'{groupname}, {run_id}'
    gallery_generator_run(save_dir, title=gallery_title)
