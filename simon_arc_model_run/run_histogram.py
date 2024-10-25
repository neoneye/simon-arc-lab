import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.histogram import Histogram

class Metric(Enum):
    EXAMPLES_HAVE_SAME_OUTPUT_HISTOGRAM = 'examples_have_same_output_histogram'

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('rearc_easy', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/easy')),
    # ('rearc_hard', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/hard')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

summary_list = []
count_correct = 0
count_incorrect = 0
count_other = 0
number_of_items_in_list = len(groupname_pathtotaskdir_list)
total_elapsed_time = 0
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    start_time = time.time()

    for task in taskset.tasks:

        input_histogram_list = []
        for i in range(task.count_examples + task.count_tests):
            h = Histogram.create_with_image(task.input_images[i])
            input_histogram_list.append(h)
        input_union, input_intersection = Histogram.union_intersection(input_histogram_list)

        output_histogram_list = []
        for i in range(task.count_examples):
            h = Histogram.create_with_image(task.output_images[i])
            output_histogram_list.append(h)
        output_union, output_intersection = Histogram.union_intersection(output_histogram_list)

        same_unique_colors_for_input_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            if input_histogram.unique_colors_set() != output_histogram.unique_colors_set():
                same_unique_colors_for_input_output = False
                break

        same_histogram_for_input_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            if input_histogram != output_histogram:
                same_histogram_for_input_output = False
                break

        same_histogram_for_all_outputs = output_union == output_intersection

        color_insert_union = set()
        color_insert_intersection = set()
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            color_insert = output_histogram.unique_colors_set() - input_histogram.unique_colors_set()
            color_insert_union = color_insert_union | color_insert
            if i == 0:
                color_insert_intersection = color_insert
            else:
                color_insert_intersection = color_insert_intersection & color_insert
        
        has_color_insert = len(color_insert_intersection) > 0

        color_insert_difference = color_insert_union - color_insert_intersection
        optional_color_insert_set = set()
        has_optional_color_insert = False
        if len(color_insert_difference) in [1, 2]:
            optional_color_insert_set = color_insert_difference
            has_optional_color_insert = True

        color_remove_union = set()
        color_remove_intersection = set()
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            color_remove = input_histogram.unique_colors_set() - output_histogram.unique_colors_set()
            color_remove_union = color_remove_union | color_remove
            if i == 0:
                color_remove_intersection = color_remove
            else:
                color_remove_intersection = color_remove_intersection & color_remove
        
        has_color_remove = len(color_remove_intersection) > 0

        output_colors_is_subset_input_colors = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            input_colors = input_histogram.unique_colors_set()
            output_colors = output_histogram.unique_colors_set()
            if output_colors.issubset(input_colors) == False:
                output_colors_is_subset_input_colors = False
                break

        for test_index in range(task.count_tests):
            input_image = task.test_input(test_index)
            output_image = task.test_output(test_index)
            input_histogram = Histogram.create_with_image(input_image)
            output_histogram = Histogram.create_with_image(output_image)

            if same_histogram_for_input_output:
                if output_histogram == input_histogram:
                    count_correct += 1
                    continue
                else:
                    count_incorrect += 1
                    print(f"same_histogram_for_input_output: {task.metadata_task_id} test={test_index}")
            
            if same_histogram_for_all_outputs:
                if output_histogram.unique_colors_set() == output_intersection:
                    count_correct += 1
                    continue
                else:
                    count_incorrect += 1
                    print(f"same_histogram_for_all_outputs: {task.metadata_task_id} test={test_index}")
            
            if same_unique_colors_for_input_output:
                if output_histogram.unique_colors_set() == input_histogram.unique_colors_set():
                    count_correct += 1
                    continue
                else:
                    count_incorrect += 1
                    print(f"same_unique_colors_for_input_output: {task.metadata_task_id} test={test_index}")
            
            if has_color_insert or has_color_remove or has_optional_color_insert:
                predicted_colors = input_histogram.unique_colors_set()
                if has_color_insert:
                    predicted_colors = predicted_colors | color_insert_intersection
                if has_color_remove:
                    predicted_colors = predicted_colors - color_remove_intersection
                if output_histogram.unique_colors_set() == predicted_colors:
                    count_correct += 1
                    continue
                if has_optional_color_insert:
                    predicted_colors2 = predicted_colors | optional_color_insert_set
                    if output_histogram.unique_colors_set() == predicted_colors2:
                        count_correct += 1
                        continue
                count_incorrect += 1
                print(f"has_color_insert/has_color_remove/has_optional_color_insert: {task.metadata_task_id} test={test_index}")

            if output_colors_is_subset_input_colors:
                if output_histogram.unique_colors_set().issubset(input_histogram.unique_colors_set()):
                    count_correct += 1
                    continue
                else:
                    count_incorrect += 1
                    print(f"output_colors_is_subset_input_colors: {task.metadata_task_id} test={test_index}")

            count_other += 1
            print(f"other: {task.metadata_task_id} test={test_index}")


    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

print(f"Correct: {count_correct}")
print(f"Incorrect: {count_incorrect}")
print(f"Other: {count_other}")
print("Summary:\n" + "\n".join(summary_list))
print(f"Total elapsed time: {total_elapsed_time:,.1f} seconds")
