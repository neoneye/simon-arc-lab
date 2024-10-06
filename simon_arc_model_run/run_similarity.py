import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_similarity import ImageSimilarity, Feature
from simon_arc_lab.task_similarity import TaskSimilarity
from simon_arc_lab.image_noise import image_noise_one_pixel

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

test_with_1pixel_noise = False
show_plot_intersectioncount = False
show_plot_testaccuracy = False

summary_list = []
number_of_items_in_list = len(groupname_pathtotaskdir_list)
total_elapsed_time = 0
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    # start time stamp, so I can measure elapsed time
    start_time = time.time()

    accumulated_test_accuracy_list = []
    accumulated_score_average_list = []
    accumulated_intersectioncount_list = []
    for task in taskset.tasks:
        # Exercise the TaskSimilarity class.
        # Measure how accurately the 'test' input/output satisfies the same features as the 'example' input/output pairs.
        test_score_list = []
        ts = TaskSimilarity.create_with_task(task)
        for i in range(task.count_tests):
            output = task.test_output(i)

            if test_with_1pixel_noise:
                output = image_noise_one_pixel(output.copy(), 0)
            
            score = ts.measure_test_prediction(output, i)
            # if score < 70:
            #     print(f"task with low score {score}")
            test_score_list.append(score)
            accumulated_test_accuracy_list.append(score)
        test_accuracy = ",".join([str(x) for x in test_score_list])

        # Exercises the ImageSimilarity class.
        score_list = []
        feature_set_intersection = set()
        for i in range(task.count_examples + task.count_tests):
            input = task.input_images[i]
            output = task.output_images[i]
            image_similarity = ImageSimilarity.create_with_images(input, output)
            score = image_similarity.jaccard_index()
            feature_list = image_similarity.get_satisfied_features()
            # feature_strings_joined = Feature.format_feature_list(feature_list)
            # print(f"pair: {i} score: {score} features: {feature_strings_joined}")

            feature_set = set(feature_list)
            if i == 0:
                feature_set_intersection = feature_set
            else:
                feature_set_intersection = feature_set & feature_set_intersection

            score_list.append(score)
        score_min = min(score_list)
        score_max = max(score_list)
        score_average = sum(score_list) / len(score_list)
        score_std_dev = np.std(score_list, ddof=1)

        count_features_set_intersection = len(feature_set_intersection)
        accumulated_intersectioncount_list.append(count_features_set_intersection)
        if score_min > 0:
            print(f"Task: '{task.metadata_task_id}'    min: {score_min} average: {score_average:,.1f} max: {score_max} std_dev: {score_std_dev:,.1f} intersection: {count_features_set_intersection}  test_accuracy: {test_accuracy}  task_summary: {ts.summary()}")
        accumulated_score_average_list.append(score_average)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

    accumulated_score_average = sum(accumulated_score_average_list) / len(accumulated_score_average_list)
    accumulated_accuracy_average = sum(accumulated_test_accuracy_list) / len(accumulated_test_accuracy_list)
    summary = f"Group: '{groupname}'    similarity: {accumulated_score_average:,.1f}   accuracy: {accumulated_accuracy_average:,.1f}  elapsed: {elapsed_time:,.1f} seconds"
    summary_list.append(summary)

    if show_plot_intersectioncount:
        plt.hist(accumulated_intersectioncount_list, bins=100)
        plt.title(f"Group: '{groupname}'")
        plt.xlabel("Intersection count")
        plt.ylabel("Frequency")
        plt.show()

    if show_plot_testaccuracy:
        plt.hist(accumulated_test_accuracy_list, bins=100)
        plt.title(f"Group: '{groupname}'")
        plt.xlabel("Test accuracy")
        plt.ylabel("Frequency")
        plt.show()

print("Summary:\n" + "\n".join(summary_list))
print(f"Total elapsed time: {total_elapsed_time:,.1f} seconds")
