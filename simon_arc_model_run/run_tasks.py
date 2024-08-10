import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.rle.serialize import *
from simon_arc_lab.rle.deserialize import *
from simon_arc_model.model import Model

def process_task(task: Task, model: Model):
    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()

    task_formatter = TaskFormatterRLECompact(task_without_test_output)
    output_ids = task_formatter.output_ids()

    dataset_name = 'SIMON-SOLVE-V1'

    for test_index in range(task.count_tests):
        input = task_formatter.to_string()
        input_image = task.test_input(test_index)
        expected_output_image = task.test_output(test_index)
        test_output_id = output_ids[task_without_test_output.count_examples + test_index]

        # Predict the entire image of the test output image
        prompt = f"{dataset_name}, {test_output_id}, predict image\n{input}"
        response = model.process(prompt)

        problem_deserialize = True
        try:
            predicted_output_image = deserialize(response)
            problem_deserialize = False
        except Exception as e:
            print(f'Error deserializing response for task {task.metadata_task_id} test={test_index}. Error: {e}')
            predicted_output_image = np.zeros((5, 5), dtype=np.uint8)

        if problem_deserialize:
            status = 'problemdeserialize'
        elif expected_output_image is None:
            status = 'unverified'
        elif np.array_equal(predicted_output_image, expected_output_image):
            status = 'correct'
        else:
            status = 'incorrect'

        title = f'{task.metadata_task_id} test={test_index} {status}'

        save_path = f'result_{task.metadata_task_id}_test{test_index}_{status}.png'
        # save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)


# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model168'
model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model179'

path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/Mini-ARC/data'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ConceptARC/data'
# path_to_task_dir = os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')
taskset = TaskSet.load_directory(path_to_task_dir)

# Load model
model = Model(model_directory, 512)

for task in taskset.tasks:
    if task.total_pixel_count() > 500:
        continue
    process_task(task, model)
