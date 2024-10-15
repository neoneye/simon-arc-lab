import os
import numpy as np
from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_lab.rle.deserialize import DeserializeError
from .predict_output_base import PredictOutputBase
from .work_item_status import WorkItemStatus

class WorkItem:
    def __init__(self, task: Task, test_index: int, refinement_step: Optional[int], predictor: PredictOutputBase):
        self.predictor_name = predictor.name()
        # print(f'WorkItem: task={task.metadata_task_id} test={test_index} predictor={self.predictor_name}')
        self.task = task
        self.test_index = test_index
        self.refinement_step = refinement_step
        self.predictor = predictor
        self.predicted_output_image = None
        self.status = WorkItemStatus.UNASSIGNED

    def process(self, context: dict):
        self.predictor.execute(context)

        task = self.task
        test_index = self.test_index
        task_id = task.metadata_task_id
        expected_output_image = task.test_output(test_index)

        try:
            self.predicted_output_image = self.predictor.predicted_image()
        except DeserializeError as e:
            # print(f'RLE decoding error for task {task_id} test={test_index}. Error: {e} score={e.score}')
            self.status = WorkItemStatus.PROBLEM_DESERIALIZE
            return
        self.assign_status()

    def assign_status(self):
        if self.predicted_output_image is None:
            self.status = WorkItemStatus.PROBLEM_MISSING_PREDICTION_IMAGE
            return
        
        expected_output_image = self.task.test_output(self.test_index)
        if expected_output_image is None:
            self.status = WorkItemStatus.UNVERIFIED
            return
        
        if np.array_equal(self.predicted_output_image, expected_output_image):
            self.status = WorkItemStatus.CORRECT
        else:
            self.status = WorkItemStatus.INCORRECT

    def show(self, save_dir_path: Optional[str] = None):
        self.show_predicted_output_image(self.predicted_output_image, save_dir_path)

    def show_predicted_output_image(self, predicted_output_image: np.array, save_dir_path: Optional[str] = None):
        task = self.task
        test_index = self.test_index
        input_image = task.test_input(test_index)
        task_id = task.metadata_task_id
        status_string = self.status.to_string()

        expected_output_image = task.test_output(test_index)

        # Human readable title
        if self.refinement_step is not None:
            title_step = f'step={self.refinement_step} '
        else:
            title_step = None
        title_items_optional = [
            task_id,
            f'test={test_index}',
            title_step,
            self.predictor_name,
            status_string,
        ]
        title_items = [item for item in title_items_optional if item is not None]
        title = ' '.join(title_items)

        # Format filename
        if self.refinement_step is not None:
            filename_step = f'step{self.refinement_step} '
        else:
            filename_step = None
        filename_items_optional = [
            task_id,
            f'test{test_index}',
            filename_step,
            self.predictor_name,
            status_string,
        ]
        filename_items = [item for item in filename_items_optional if item is not None]
        filename = '_'.join(filename_items) + '.png'

        if save_dir_path is not None:
            save_path = os.path.join(save_dir_path, filename)
        else:
            save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)
