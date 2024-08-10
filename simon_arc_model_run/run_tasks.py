import os
import sys
from tqdm import tqdm
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import *
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_model.model import Model
from simon_arc_model.predict_output_v1 import PredictOutputV1

class WorkItem:
    def __init__(self, task: Task, test_index: int):
        self.task = task
        self.test_index = test_index
        self.predictor = PredictOutputV1(task, test_index)
        self.predicted_output_image = None

    @classmethod
    def collect_predictions_as_arcprize2024_submission_dict(cls, work_items: list['WorkItem']) -> dict:
        result_dict = {}
        for work_item in work_items:
            if work_item.predicted_output_image is None:
                continue
            task_id = work_item.task.metadata_task_id

            # Create a new entry in the result_dict if it doesn't exist, with dummy images
            # This is in order to handle tasks that have 2 or more test pairs.
            if task_id not in result_dict:
                count_tests = work_item.task.count_tests
                dummy_image = [[0]]
                attempts_dict = {
                    'attempt_1': dummy_image
                }
                test_list = []
                for _ in range(count_tests):
                    test_list.append(attempts_dict)
                result_dict[task_id] = test_list

            # Update the existing entry in the result_dict with the predicted image
            image = work_item.predicted_output_image.tolist()
            result_dict[task_id][work_item.test_index]['attempt_1'] = image
        return result_dict
    
    @classmethod
    def save_arcprize2024_submission_file(cls, work_items: list['WorkItem'], path_to_json_file: str):
        dict = cls.collect_predictions_as_arcprize2024_submission_dict(work_items)
        with open(path_to_json_file, 'w') as f:
            json.dump(dict, f)

    def process(self, model: Model):
        self.predictor.execute(model)

        task = self.task
        test_index = self.test_index
        input_image = task.test_input(test_index)
        expected_output_image = task.test_output(test_index)

        problem_deserialize = True
        try:
            predicted_output_image = self.predictor.predicted_image()
            self.predicted_output_image = predicted_output_image
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

        save_path = f'run_tasks_result/{task.metadata_task_id}_test{test_index}_{status}.png'
        save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)

        if status == 'correct':
            print(f'Correct prediction for task {task.metadata_task_id} test={test_index}')

class WorkManager:
    def __init__(self, model: Model, taskset: TaskSet):
        self.model = model
        self.taskset = taskset
        self.work_items = []

    def load_work_items(self):
        for task in self.taskset.tasks:
            for test_index in range(task.count_tests):
                work_item = WorkItem(task, test_index)
                self.work_items.append(work_item)

    def discard_items_with_too_long_prompts(self, max_prompt_length: int):
        """
        Ignore those where the prompt longer than what the model can handle.
        """
        count_before = len(self.work_items)
        filtered_work_items = []
        for work_item in self.work_items:
            if len(work_item.predictor.prompt()) <= max_prompt_length:
                filtered_work_items.append(work_item)
        count_after = len(filtered_work_items)
        self.work_items = filtered_work_items
        print(f'Removed {count_before - count_after} work items with too long prompt. Remaining are {count_after} work items.')

    def process_all_work_items(self):
        for work_item in tqdm(self.work_items, desc="Processing work items"):
            work_item.process(self.model)

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model168'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model179'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model180'
# model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model181'
model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model182'

path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/training'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data/evaluation'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/Mini-ARC/data'
# path_to_task_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ConceptARC/data'
path_to_task_dir = os.path.join(PROJECT_ROOT, 'testdata', 'simple_arc_tasks')
taskset = TaskSet.load_directory(path_to_task_dir)


# Load model
model = Model(model_directory, 512)

wm = WorkManager(model, taskset)
wm.load_work_items()
wm.discard_items_with_too_long_prompts(500)
wm.process_all_work_items()

WorkItem.save_arcprize2024_submission_file(wm.work_items, 'submission.json')
