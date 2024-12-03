import os
import numpy as np
from typing import Optional
from simon_arc_lab.task import Task
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_lab.rle.deserialize import DeserializeError
from .predict_output_base import PredictOutputBase
from .work_item import WorkItem
from .work_item_status import WorkItemStatus

class WorkItemWithPredictor(WorkItem):
    def __init__(self, task: Task, test_index: int, refinement_step: Optional[int], predictor: PredictOutputBase):
        super().__init__(task, test_index)
        self.predictor_name = predictor.name()
        # print(f'WorkItem: task={task.metadata_task_id} test={test_index} predictor={self.predictor_name}')
        self.refinement_step = refinement_step
        self.predictor = predictor

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

    def title_fields(self) -> list:
        if self.refinement_step is not None:
            title_step = f'step={self.refinement_step}'
        else:
            title_step = None
        return [
            self.task.metadata_task_id,
            f'test={self.test_index}',
            title_step,
            self.predictor_name,
            self.status.to_string(),
        ]
    
    def filename_fields(self) -> list:
        if self.refinement_step is not None:
            filename_step = f'step{self.refinement_step}'
        else:
            filename_step = None
        return [
            self.task.metadata_task_id,
            f'test{self.test_index}',
            filename_step,
            self.predictor_name,
            self.status.to_string(),
        ]
