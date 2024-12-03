from typing import Optional
from simon_arc_lab.task import Task
from .work_item import WorkItem

class WorkItemWithRefinementStep(WorkItem):
    def __init__(self, task: Task, test_index: int, refinement_step: Optional[int]):
        super().__init__(task, test_index)
        self.refinement_step = refinement_step

    def title_fields(self) -> list:
        if self.refinement_step is not None:
            title_step = f'step={self.refinement_step}'
        else:
            title_step = None
        return [
            self.task.metadata_task_id,
            f'test={self.test_index}',
            title_step,
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
            self.status.to_string(),
        ]
