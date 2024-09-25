import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_similarity import ImageSimilarity

model_iteration = 530

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    ('rearc_easy', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/easy')),
    ('rearc_hard', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/hard')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

summary_list = []
number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    accumulated_score_average_list = []
    for task in taskset.tasks:
        score_list = []
        for i in range(task.count_examples + task.count_tests):
            input = task.input_images[i]
            output = task.output_images[i]
            score = ImageSimilarity(input, output).jaccard_index()
            # print(f"pair: {i} score: {score}")
            score_list.append(score)
        score_min = min(score_list)
        score_max = max(score_list)
        score_average = sum(score_list) / len(score_list)
        score_std_dev = np.std(score_list, ddof=1)
        # print(f"Task: '{task.metadata_task_id}'    min: {score_min} average: {score_average:,.1f} max: {score_max} std_dev: {score_std_dev:,.1f}")
        accumulated_score_average_list.append(score_average)

    accumulated_score_average = sum(accumulated_score_average_list) / len(accumulated_score_average_list)
    summary = f"Group: '{groupname}'    similarity: {accumulated_score_average:,.1f}"
    summary_list.append(summary)

print("Summary:\n" + "\n".join(summary_list))
