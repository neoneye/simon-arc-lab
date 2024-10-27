import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.task_color_profile import BenchmarkTaskColorProfile

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    # ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('diva', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-diva/data')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('rearc_easy', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/easy')),
    # ('rearc_hard', os.path.join(path_to_arc_dataset_collection_dataset, 'RE-ARC/data/hard')),
    # ('sortofarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Sort-of-ARC/data')),
    # ('synth_riddles', os.path.join(path_to_arc_dataset_collection_dataset, 'synth_riddles/data')),
    # ('Sequence_ARC', os.path.join(path_to_arc_dataset_collection_dataset, 'Sequence_ARC/data')),
    # ('PQA', os.path.join(path_to_arc_dataset_collection_dataset, 'PQA/data')),
    # ('nosound', os.path.join(path_to_arc_dataset_collection_dataset, 'nosound/data')),
    # ('dbigham', os.path.join(path_to_arc_dataset_collection_dataset, 'dbigham/data')),
    # ('arc-community', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-community/data')),
    # ('IPARC', os.path.join(path_to_arc_dataset_collection_dataset, 'IPARC/data')),
    ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

invalid_task_id_list = [
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

benchmark = BenchmarkTaskColorProfile(verbose=False)
number_of_items_in_list = len(groupname_pathtotaskdir_list)
total_elapsed_time = 0
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    start_time = time.time()

    for task in taskset.tasks:
        if task.metadata_task_id in invalid_task_id_list:
            continue
        benchmark.measure_task(task)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

benchmark.print_summary()

print(f"\nTotal elapsed time: {total_elapsed_time:,.1f} seconds")
