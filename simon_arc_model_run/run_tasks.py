import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_model.model import Model
from simon_arc_model.work_manager import WorkManager

# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model229' # training=21 evaluation=6
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model231'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model232'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model233'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model235'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model237'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model239'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model240' # training=23 evaluation=4
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model241'
model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model242'

path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/Mini-ARC/data'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ConceptARC/data'
# path_to_task_dir = os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')
taskset = TaskSet.load_directory(path_to_task_dir)


# Load model
model = Model(model_directory, 512)

wm = WorkManager(model, taskset)
wm.discard_items_with_too_long_prompts(500)
# wm.process_all_work_items()
wm.process_all_work_items(save_dir='run_tasks_result')
# wm.process_all_work_items(show=True)
wm.summary()
wm.save_arcprize2024_submission_file('submission.json')
