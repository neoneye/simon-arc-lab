import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.histogram import Histogram

class Metric(Enum):
    SAME_HISTOGRAM_FOR_INPUT_OUTPUT = 'same_histogram_for_input_output'
    SAME_HISTOGRAM_FOR_ALL_OUTPUTS = 'same_histogram_for_all_outputs'
    SAME_UNIQUE_COLORS_FOR_INPUT_OUTPUT = 'same_unique_colors_for_input_output'
    SAME_INSERT_REMOVE = 'same_insert_remove'
    OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS = 'output_colors_is_subset_input_colors'
    OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS_WITH_INSERT_REMOVE = 'output_colors_is_subset_input_colors_with_insert_remove'
    COLOR_MAPPING = 'color_mapping'
    OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION = 'output_colors_is_subset_example_output_union'
    OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OUTPUTINTERSECTIONCOLORS = 'output_colors_is_subset_inputcolors_union_outputintersectioncolors'
    OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OPTIONALOUTPUTINTERSECTIONCOLORS = 'output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors'
    MOST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'most_popular_colors_of_input_are_present_in_output'
    MOST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'most_popular_colors_of_input_are_not_present_in_output'
    LEAST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'least_popular_colors_of_input_are_present_in_output'
    LEAST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'least_popular_colors_of_input_are_not_present_in_output'
    INBETWEEN_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'inbetween_colors_of_input_are_present_in_output'
    INBETWEEN_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'inbetween_colors_of_input_are_not_present_in_output'

    def format_with_value(self, value: int) -> str:
        suffix = ''
        if self == Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS:
            suffix = ' weak'
        elif self == Metric.OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION:
            suffix = ' weak'
        return f"{self.name.lower()}: {value}{suffix}"

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    ('diva', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-diva/data')),
    ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    ('rearc_easy', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/easy')),
    ('rearc_hard', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/hard')),
    ('sortofarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Sort-of-ARC/data')),
    ('synth_riddles', os.path.join(path_to_arc_dataset_collection_dataset, 'synth_riddles/data')),
    ('Sequence_ARC', os.path.join(path_to_arc_dataset_collection_dataset, 'Sequence_ARC/data')),
    ('PQA', os.path.join(path_to_arc_dataset_collection_dataset, 'PQA/data')),
    ('nosound', os.path.join(path_to_arc_dataset_collection_dataset, 'nosound/data')),
    ('dbigham', os.path.join(path_to_arc_dataset_collection_dataset, 'dbigham/data')),
    ('arc-community', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-community/data')),
    ('IPARC', os.path.join(path_to_arc_dataset_collection_dataset, 'IPARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

invalid_task_id_list = [
    # ARC-AGI
    'a8610ef7',
    # IPARC
    'CatB_Hard_Task005',
    'CatB_Hard_Task017',
    'CatB_Hard_Task019',
    'CatB_Hard_Task089',
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

count_correct = defaultdict(int)
count_incorrect = defaultdict(int)
count_correct_subitem = defaultdict(int)
count_incorrect_subitem = defaultdict(int)
count_issue = 0
number_of_items_in_list = len(groupname_pathtotaskdir_list)
total_elapsed_time = 0
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    start_time = time.time()

    for task in taskset.tasks:
        if task.metadata_task_id in invalid_task_id_list:
            continue

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
            predicted_colors = input_histogram.unique_colors_set()
            output_colors = output_histogram.unique_colors_set()
            if output_colors.issubset(predicted_colors) == False:
                output_colors_is_subset_input_colors = False
                break

        # Determines if the output colors are a subset of the input colors with insert/remove
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9565186b
        output_colors_is_subset_input_colors_with_insert_remove = has_color_insert or has_color_remove
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            input_colors = input_histogram.unique_colors_set()
            predicted_colors = (input_colors | color_insert_intersection) - color_remove_intersection
            if len(predicted_colors) == 0:
                output_colors_is_subset_input_colors_with_insert_remove = False
                break
            output_colors = output_histogram.unique_colors_set()
            if output_colors.issubset(predicted_colors) == False:
                output_colors_is_subset_input_colors_with_insert_remove = False
                break

        output_colors_is_subset_example_output_union = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            output_colors = output_histogram.unique_colors_set()
            if output_colors.issubset(output_union) == False:
                output_colors_is_subset_example_output_union = False
                break

        # Determines if the output colors are a subset of (input_colors UNION example_output_intersection)
        # https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=Center4
        # https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=ExtractObjects6
        # https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=FilledNotFilled7
        # https://neoneye.github.io/arc/edit.html?dataset=Mini-ARC&task=find_the_most_frequent_color_for_every_2x2_l6ad7ge3gc5rtysj7p
        output_colors_is_subset_inputcolors_union_outputintersectioncolors = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            predicted_colors = input_histogram.unique_colors_set() | output_intersection
            output_colors = output_histogram.unique_colors_set()
            if output_colors.issubset(predicted_colors) == False:
                output_colors_is_subset_inputcolors_union_outputintersectioncolors = False
                break

        # Determines if the output colors are a subset of (input_colors UNION optional_output_intersection)
        # https://neoneye.github.io/arc/edit.html?dataset=RE-ARC-easy&task=f2829549
        output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors = False
        if has_optional_color_insert:
            output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors = True
            for i in range(task.count_examples):
                input_histogram = input_histogram_list[i]
                output_histogram = output_histogram_list[i]
                predicted_colors = input_histogram.unique_colors_set() | optional_color_insert_set
                output_colors = output_histogram.unique_colors_set()
                if output_colors.issubset(predicted_colors) == False:
                    output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors = False
                    break

        # Determines if there a color mapping between input and output histograms
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6ea4a07e
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=913fb3ed
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=54d9e175
        color_mapping = {}
        has_color_mapping = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            key = input_histogram.unique_colors_pretty()
            value = output_histogram.unique_colors_set()
            if key in color_mapping:
                if color_mapping[key] != value:
                    has_color_mapping = False
                    break
            color_mapping[key] = value
        if has_color_mapping == False:
            color_mapping = None

        # Determines if the most popular colors of the input gets passed onto the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9565186b
        most_popular_colors_of_input_are_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = set(input_histogram.most_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors) == False:
                most_popular_colors_of_input_are_present_in_output = False
                break

        # Determines if the most popular colors of the input are not present in the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9b4f6fc
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ca8de6ea
        most_popular_colors_of_input_are_not_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = set(input_histogram.most_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors):
                most_popular_colors_of_input_are_not_present_in_output = False
                break

        # Determines if the least popular colors of the input are present in the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=50aad11f
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5289ad53
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5a5a2103
        least_popular_colors_of_input_are_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = set(input_histogram.least_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors) == False:
                least_popular_colors_of_input_are_present_in_output = False
                break
        
        # Determines if the least popular colors of the input are not present in the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=0a2355a6
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=37d3e8b2
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=604001fa
        least_popular_colors_of_input_are_not_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = set(input_histogram.least_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors):
                least_popular_colors_of_input_are_not_present_in_output = False
                break

        # Determines if the inbetween colors of the input are present in the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3ee1011a
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=93b4f4b3
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=94133066
        inbetween_colors_of_input_are_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
            if len(special_colors) == 0:
                inbetween_colors_of_input_are_present_in_output = False
                continue
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors) == False:
                inbetween_colors_of_input_are_present_in_output = False
                break
        
        # Determines if the inbetween colors of the input are not present in the output
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=aabf363d
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ddf7fa4f
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cf98881b
        # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7c008303
        inbetween_colors_of_input_are_not_present_in_output = True
        for i in range(task.count_examples):
            input_histogram = input_histogram_list[i]
            output_histogram = output_histogram_list[i]
            special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
            if len(special_colors) == 0:
                inbetween_colors_of_input_are_not_present_in_output = False
                continue
            output_colors = output_histogram.unique_colors_set()
            if special_colors.issubset(output_colors):
                inbetween_colors_of_input_are_not_present_in_output = False
                break
        
        for test_index in range(task.count_tests):
            input_image = task.test_input(test_index)
            output_image = task.test_output(test_index)
            input_histogram = Histogram.create_with_image(input_image)
            output_histogram = Histogram.create_with_image(output_image)

            if same_histogram_for_input_output:
                if output_histogram == input_histogram:
                    count_correct[Metric.SAME_HISTOGRAM_FOR_INPUT_OUTPUT] += 1
                    continue
                count_incorrect[Metric.SAME_HISTOGRAM_FOR_INPUT_OUTPUT] += 1
                # print(f"same_histogram_for_input_output: {task.metadata_task_id} test={test_index}")
            
            if same_histogram_for_all_outputs:
                if output_histogram.unique_colors_set() == output_intersection:
                    count_correct[Metric.SAME_HISTOGRAM_FOR_ALL_OUTPUTS] += 1
                    continue
                count_incorrect[Metric.SAME_HISTOGRAM_FOR_ALL_OUTPUTS] += 1
                # print(f"same_histogram_for_all_outputs: {task.metadata_task_id} test={test_index}")
            
            if same_unique_colors_for_input_output:
                if output_histogram.unique_colors_set() == input_histogram.unique_colors_set():
                    count_correct[Metric.SAME_UNIQUE_COLORS_FOR_INPUT_OUTPUT] += 1
                    continue
                count_incorrect[Metric.SAME_UNIQUE_COLORS_FOR_INPUT_OUTPUT] += 1
                # print(f"same_unique_colors_for_input_output: {task.metadata_task_id} test={test_index}")
            
            if most_popular_colors_of_input_are_present_in_output:
                special_colors = set(input_histogram.most_popular_color_list())
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors):
                    count_correct_subitem[Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1

            if most_popular_colors_of_input_are_not_present_in_output:
                special_colors = set(input_histogram.most_popular_color_list())
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors) == False:
                    count_correct_subitem[Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1

            if least_popular_colors_of_input_are_present_in_output:
                special_colors = set(input_histogram.least_popular_color_list())
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors):
                    count_correct_subitem[Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1

            if least_popular_colors_of_input_are_not_present_in_output:
                special_colors = set(input_histogram.least_popular_color_list())
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors) == False:
                    count_correct_subitem[Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1

            if inbetween_colors_of_input_are_present_in_output:
                special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors):
                    count_correct_subitem[Metric.INBETWEEN_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.INBETWEEN_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT] += 1

            if inbetween_colors_of_input_are_not_present_in_output:
                special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
                output_colors = output_histogram.unique_colors_set()
                if special_colors.issubset(output_colors) == False:
                    count_correct_subitem[Metric.INBETWEEN_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1
                else:
                    count_incorrect_subitem[Metric.INBETWEEN_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT] += 1

            if has_color_insert or has_color_remove or has_optional_color_insert:
                predicted_colors = input_histogram.unique_colors_set()
                if has_color_insert:
                    predicted_colors = predicted_colors | color_insert_intersection
                if has_color_remove:
                    predicted_colors = predicted_colors - color_remove_intersection
                if output_histogram.unique_colors_set() == predicted_colors:
                    count_correct[Metric.SAME_INSERT_REMOVE] += 1
                    continue
                if has_optional_color_insert:
                    predicted_colors2 = predicted_colors | optional_color_insert_set
                    if output_histogram.unique_colors_set() == predicted_colors2:
                        count_correct[Metric.SAME_INSERT_REMOVE] += 1
                        continue
                count_incorrect[Metric.SAME_INSERT_REMOVE] += 1
                # print(f"has_color_insert/has_color_remove/has_optional_color_insert: {task.metadata_task_id} test={test_index}")

            if output_colors_is_subset_input_colors:
                if output_histogram.unique_colors_set().issubset(input_histogram.unique_colors_set()):
                    count_correct[Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS] += 1
                    continue
                count_incorrect[Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS] += 1
                # print(f"output_colors_is_subset_input_colors: {task.metadata_task_id} test={test_index}")

            if output_colors_is_subset_input_colors_with_insert_remove:
                input_colors = input_histogram.unique_colors_set()
                predicted_colors = (input_colors | color_insert_intersection) - color_remove_intersection
                if output_histogram.unique_colors_set().issubset(predicted_colors):
                    count_correct[Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS_WITH_INSERT_REMOVE] += 1
                    continue
                count_incorrect[Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS_WITH_INSERT_REMOVE] += 1
                # print(f"output_colors_is_subset_input_colors_with_insert_remove: {task.metadata_task_id} test={test_index}")

            if output_colors_is_subset_inputcolors_union_outputintersectioncolors:
                predicted_colors = input_histogram.unique_colors_set() | output_intersection
                if output_histogram.unique_colors_set().issubset(predicted_colors):
                    count_correct[Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OUTPUTINTERSECTIONCOLORS] += 1
                    continue
                count_incorrect[Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OUTPUTINTERSECTIONCOLORS] += 1
                # print(f"output_colors_is_subset_input_union_example_output_intersection: {task.metadata_task_id} test={test_index}")

            if has_color_mapping:
                key = input_histogram.unique_colors_pretty()
                if key in color_mapping and output_histogram.unique_colors_set() == color_mapping[key]:
                    count_correct[Metric.COLOR_MAPPING] += 1
                    continue
                count_incorrect[Metric.COLOR_MAPPING] += 1
                # print(f"has_color_mapping: {task.metadata_task_id} test={test_index}")
            
            if output_colors_is_subset_example_output_union:
                if output_histogram.unique_colors_set().issubset(output_union):
                    count_correct[Metric.OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION] += 1
                    continue
                count_incorrect[Metric.OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION] += 1
                # print(f"output_colors_is_subset_example_output_union: {task.metadata_task_id} test={test_index}")

            if output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors:
                predicted_colors = input_histogram.unique_colors_set() | optional_color_insert_set
                if output_histogram.unique_colors_set().issubset(predicted_colors):
                    count_correct[Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OPTIONALOUTPUTINTERSECTIONCOLORS] += 1
                    continue
                count_incorrect[Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OPTIONALOUTPUTINTERSECTIONCOLORS] += 1
                #print(f"output_colors_is_subset_input_union_optional_output_intersection: {task.metadata_task_id} test={test_index}")

            count_issue += 1
            print(f"issue: {task.metadata_task_id} test={test_index}")
            # most popular color manipulation
            # https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9565186b


    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

print(f"\nIssues: {count_issue}, puzzles where the transformation couldn't be identified.")

print(f"\nCorrect:")
sorted_counters = sorted(count_correct.items(), key=lambda x: (-x[1], x[0].name))
for key, count in sorted_counters:
    s = key.format_with_value(count)
    print(f"  {s}")

print(f"\nIncorrect, where the check was triggered, but not satisfied, a false positive:")
sorted_counters = sorted(count_incorrect.items(), key=lambda x: (-x[1], x[0].name))
for key, count in sorted_counters:
    s = key.format_with_value(count)
    print(f"  {s}")

print(f"\nCorrect subitems:")
sorted_counters = sorted(count_correct_subitem.items(), key=lambda x: (-x[1], x[0].name))
for key, count in sorted_counters:
    s = key.format_with_value(count)
    print(f"  {s}")

print(f"\nIncorrect subitems, where the check was triggered, but not satisfied, a false positive:")
sorted_counters = sorted(count_incorrect_subitem.items(), key=lambda x: (-x[1], x[0].name))
for key, count in sorted_counters:
    s = key.format_with_value(count)
    print(f"  {s}")

print(f"\nTotal elapsed time: {total_elapsed_time:,.1f} seconds")
