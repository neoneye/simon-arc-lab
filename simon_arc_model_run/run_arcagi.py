import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.rle.serialize import *
from simon_arc_model.model import Model
from simon_arc_model.runner import Runner

def dataset_items_with_task(task: Task) -> list[dict]:
    task_id = task.metadata_task_id

    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()
    # print(task_without_test_output.to_arcagi1_json(compact=True))

    task_formatter = TaskFormatterRLECompact(task_without_test_output)
    output_ids = task_formatter.output_ids()

    dataset_name = 'SIMON-SOLVE-V1'

    dataset_items = []

    # Predict the entire image of the test output image
    for test_index in range(task.count_tests):
        input = task_formatter.to_string()
        expected_output = task.test_output(test_index)
        test_output_id = output_ids[task_without_test_output.count_examples + test_index]

        instruction = f"{dataset_name}, {test_output_id}, predict image"

        output = serialize(expected_output)
        benchmark_id = f'dataset={task_id} predict=image test_index={test_index}'

        dataset_item = {
            'instruction': instruction,
            'input': input,
            'output': output,
            'benchmark': benchmark_id
        }
        dataset_items.append(dataset_item)
    
    return dataset_items


# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model151' # best so far
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model168'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model180'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model181'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model182'
model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model184'

# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
path_to_task_dir = os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')
taskset = TaskSet.load_directory(path_to_task_dir)

dataset_items = []
for task in taskset.tasks:
    if task.total_pixel_count() > 500:
        continue
    dataset_items += dataset_items_with_task(task)
print(f"Generated {len(dataset_items)} dataset items")

# Load model
model = Model(model_directory, 512)

# Initialize runner
runner = Runner(model)
runner.run(dataset_items)
