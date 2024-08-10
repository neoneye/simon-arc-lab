from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.rle.deserialize import *
from simon_arc_model.model import Model

class PredictOutputV1:
    def __init__(self, task: Task, test_index: int):
        self.task = task
        self.test_index = test_index
        self.cached_prompt = None
        self.cached_response = None

    def prompt(self) -> str:
        if self.cached_prompt:
            return self.cached_prompt

        task_without_test_output = self.task.clone()
        task_without_test_output.set_all_test_outputs_to_none()
        task_formatter = TaskFormatterRLECompact(task_without_test_output)
        test_output_id = task_formatter.test_output_id(self.test_index)
        input = task_formatter.to_string()
        prompt = f"SIMON-SOLVE-V1, {test_output_id}, predict image\n{input}"

        self.cached_prompt = prompt
        return prompt

    def execute(self, model: Model):
        if self.cached_response is not None:
            return
        prompt = self.prompt()
        response = model.process(prompt)
        self.cached_response = response

    def predicted_image(self) -> np.array:
        if self.cached_response is None:
            raise ValueError("No response cached. Call execute() first.")
        image = deserialize(self.cached_response)
        return image
