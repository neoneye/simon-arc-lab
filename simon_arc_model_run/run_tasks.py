import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_model.model import Model
from simon_arc_model.work_manager import WorkManager

# A power of 2 value, and the max length of the input prompt
CONTEXT_SIZE_LIMIT = (512, 500)
# CONTEXT_SIZE_LIMIT = (1024, 1000)

# iteration=229 training=21 evaluation=6 total=27
# iteration=240 training=23 evaluation=4 total=27
# iteration=254 training=23 evaluation=4 total=27
# iteration=255 training=23 evaluation=4 total=27
# iteration=256 training=22 evaluation=6 total=28
# iteration=262 training=23 evaluation=5 total=28
# iteration=289 training=24 evaluation=6 total=30 <-- best total
# iteration=294 training=22 evaluation=6 total=28
# iteration=309 training=24 evaluation=6 total=30 <-- best total
# iteration=351 training=20 evaluation=7 total=27 <-- best evaluation
# iteration=364 training=20 evaluation=7 total=27 <-- best evaluation
model_iteration = 421
model_name = f'simon-arc-lab-model{model_iteration}'
model_directory = f'/Users/neoneye/nobackup/git/{model_name}'

# check if the model is a dir in the file system
if not os.path.isdir(model_directory):
    print(f"Model directory '{model_directory}' does not exist.")
    sys.exit(1)

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

context_length, max_prompt_length = CONTEXT_SIZE_LIMIT
print(f"context length: {context_length}, max prompt length: {max_prompt_length}")

# Load model
model = Model(model_directory, context_length)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{model_iteration}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    wm = WorkManager(model, taskset)
    wm.discard_items_with_too_long_prompts(max_prompt_length)
    # wm.process_all_work_items()
    wm.process_all_work_items(save_dir=save_dir)
    # wm.process_all_work_items(show=True)
    wm.summary()

    gallery_title = f'{groupname}, model {model_iteration}'
    gallery_generator_run(save_dir, title=gallery_title)

    # wm.save_arcprize2024_submission_file('submission.json')
