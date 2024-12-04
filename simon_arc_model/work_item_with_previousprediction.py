import numpy as np
from simon_arc_lab.task import Task
from .work_item import WorkItem

class WorkItemWithPreviousPrediction(WorkItem):
    def __init__(self, task: Task, test_index: int, previous_predicted_output_image: np.array, unique_id: str):
        super().__init__(task, test_index)
        self.previous_predicted_output_image = previous_predicted_output_image
        self.unique_id = unique_id

    def title_fields(self) -> list:
        return [
            self.task.metadata_task_id,
            f'test={self.test_index}',
            f'unique_id={self.unique_id}',
            self.status.to_string(),
        ]
    
    def filename_fields(self) -> list:
        return [
            self.task.metadata_task_id,
            f'test{self.test_index}',
            f'uniqueid{self.unique_id}',
            self.status.to_string(),
        ]
