from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.rle.deserialize import *
from simon_arc_model.model import Model

def predict_output_v1(model: Model, task: Task, test_index: int) -> np.array:
    """
    Predict the output image for a specific test in the task.
    """
    task_without_test_output = task.clone()
    task_without_test_output.set_all_test_outputs_to_none()

    task_formatter = TaskFormatterRLECompact(task_without_test_output)
    test_output_id = task_formatter.test_output_id(test_index)
    input = task_formatter.to_string()
    prompt = f"SIMON-SOLVE-V1, {test_output_id}, predict image\n{input}"
    response = model.process(prompt)
    predicted_output_image = deserialize(response)
    return predicted_output_image

class PredictOutputV1:
    def __init__(self, task: Task, test_index: int):
        self.task = task
        self.test_index = test_index
        self.cached_prompt = None

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

    def execute(self, model: Model) -> np.array:
        prompt = self.prompt()
        response = model.process(prompt)
        predicted_output_image = deserialize(response)
        return predicted_output_image
