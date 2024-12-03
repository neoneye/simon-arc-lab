import numpy as np
from simon_arc_lab.task import Task
from .work_item import WorkItem

class WorkItemWithPreviousPrediction(WorkItem):
    def __init__(self, task: Task, test_index: int, previous_predicted_output_image: np.array):
        super().__init__(task, test_index)
        self.previous_predicted_output_image = previous_predicted_output_image
