# Assumption:
# I don't want the model to be concerned about predicting the size of the image.
# so the correct output size has to be determined in advance, before using the model.
from datetime import datetime
import os
import sys
import numpy as np
import random
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from enum import Enum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images
from simon_arc_lab.image_noise import *
from simon_arc_lab.remap import remap
from simon_arc_model.random_sampling import *

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")
import torch
import random
import numpy as np
from transformer_classifier import TransformerClassifier

model_path = '/Users/neoneye/git/python_arc/run_tasks_result/transformer_classifier.pth'

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Specify device
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = 'cuda' 
device_name = 'mps'

device = torch.device(device_name)
print(f"Device: {device}")

# Initialize and load the trained model
model = TransformerClassifier(src_vocab_size=528, num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

def pad_sequence(sequence, max_length, pad_value=0):
    if len(sequence) < max_length:
        sequence = sequence + [pad_value] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

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

    task_hash = task.metadata_task_id.__hash__()

    # builder_cls = BuilderList
    builder_cls = BuilderWithVocabulary

    xs, ys, extra = xs_ys_from_input_target_pairs(input_target_pairs, task_hash, builder_cls)

    if False:
        print(f"task: {task.metadata_task_id} row count: {len(xs)}")

        filename = f'{task.metadata_task_id}_train.jsonl'
        jsonl_path = os.path.join(save_dir, filename)
        with open(jsonl_path, 'w') as f:
            for i in range(len(xs)):
                dict = {
                    "xs": xs[i],
                    "ys": ys[i],
                    "extra": extra[i],
                }
                json_str = json.dumps(dict, separators=(',', ':'))
                f.write(json_str + "\n")
        return (0, 0)

    # clf = DecisionTreeClassifier(random_state=42)
    # clf.fit(xs, ys)

    xs2, ys2, extra2 = xs_ys_from_input_target_pairs(input_target_pairs_one_test, task_hash, builder_cls)
    # for i in range(len(xs2)):
    #     x_values = xs2[i]
    #     y_value = ys2[i]
    #     print(f"xs2[{i}]: {x_values} ys2[{i}]: {y_value}")

    #     if i > 10:
    #         break
    # predicted_values = clf.predict(xs2)

    # Prepare all input sequences
    max_length = 42  # Adjust if necessary

    # Adjust batch size based on your hardware
    batch_size = 128

    # Initialize list to collect predictions
    predicted_values = []

    # Total number of sequences
    num_sequences = len(xs2)

    # Process sequences in batches
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_sequences = xs2[start_idx:end_idx]
        
        # If sequences are already of same length, you can skip padding
        # Pad sequences if necessary
        padded_sequences = [pad_sequence(seq, max_length) for seq in batch_sequences]
        
        # Convert to tensor and move to device
        src = torch.tensor(padded_sequences, dtype=torch.long).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(src)
        
        # Get predictions
        batch_predictions = torch.argmax(output, dim=1).cpu().numpy()
        predicted_values.extend(batch_predictions)

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
    # print(f"task: {task.metadata_task_id} correct: {pred_count_correct} incorrect: {pred_count_incorrect}")

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
            extra2_item = extra2[i]
            target_pair_id = extra2_item[1]
            if target_pair_id != task.count_examples:
                continue
            target_x = extra2_item[2]
            target_y = extra2_item[3]
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


print("Vocab size: ", Vocabulary.VOCAB_SIZE)

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
