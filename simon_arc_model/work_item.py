import os
import numpy as np
from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.show_prediction_result import show_prediction_result
from .work_item_status import WorkItemStatus

class WorkItem:
    def __init__(self, task: Task, test_index: int):
        # print(f'WorkItem: task={task.metadata_task_id} test={test_index}')
        self.task = task
        self.test_index = test_index
        self.predicted_output_image = None
        self.status = WorkItemStatus.UNASSIGNED

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

    def title_fields(self) -> list:
        """
        Optional fields to include in the title of the visualization.
        When the field is None, it is not included in the title.

        Override this function if you want to customize the title.
        """
        return [
            self.task.metadata_task_id,
            f'test={self.test_index}',
            self.status.to_string(),
        ]
    
    def filename_fields(self) -> list:
        """
        Optional fields to include in the filename of the visualization.
        When the field is None, it is not included in the filename.

        Override this function if you want to customize the filename.
        """
        return [
            self.task.metadata_task_id,
            f'test{self.test_index}',
            self.status.to_string(),
        ]
    
    def resolve_title(self) -> str:
        """
        Human readable title for the visualization.
        """
        title_items_optional = self.title_fields()
        title_items = [item for item in title_items_optional if item is not None]
        return ' '.join(title_items)
    
    def resolve_filename(self) -> str:
        """
        Human readable filename for the visualization png image.
        """
        filename_items_optional = self.filename_fields()
        filename_items = [item for item in filename_items_optional if item is not None]
        return '_'.join(filename_items) + '.png'

    def show(self, save_dir_path: Optional[str] = None):
        self.show_predicted_output_image(self.predicted_output_image, save_dir_path)

    def show_predicted_output_image(self, predicted_output_image: np.array, save_dir_path: Optional[str] = None):
        task = self.task
        test_index = self.test_index
        input_image = task.test_input(test_index)

        expected_output_image = task.test_output(test_index)

        title = self.resolve_title()
        filename = self.resolve_filename()

        if save_dir_path is not None:
            save_path = os.path.join(save_dir_path, filename)
        else:
            save_path = None
        show_prediction_result(input_image, predicted_output_image, expected_output_image, title, show_grid=True, save_path=save_path)
