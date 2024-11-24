from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.task_mutator import *
from simon_arc_lab.image_distort import *
from simon_arc_lab.image_noise import *
from simon_arc_lab.image_string_representation import image_from_string
from simon_arc_model.model import Model, ModelProcessMode
from .predict_output_base import PredictOutputBase

class PredictOutputV3(PredictOutputBase):
    def __init__(self, task: Task, test_index: int, task_mutator_class: type):
        if not issubclass(task_mutator_class, TaskMutatorBase):
            raise TypeError(f"{task_mutator_class.__name__} must be a subclass of TaskMutatorBase")
        
        self.task = task
        self.test_index = test_index
        self.task_mutator = task_mutator_class(task)
        self.cached_prompt = None
        self.cached_response = None

    def prompt(self) -> str:
        if self.cached_prompt:
            return self.cached_prompt

        task_without_test_output = self.task_mutator.transformed_task()
        task_without_test_output.set_all_test_outputs_to_none()

        task_formatter = TaskFormatterRLECompact(task_without_test_output)
        test_output_id = task_formatter.test_output_id(self.test_index)
        input = task_formatter.to_string()
        prompt = f"SIMON-SOLVE-V1, {test_output_id}, predict raw pixels\n{input}"

        self.cached_prompt = prompt
        return prompt

    def execute(self, context: dict):
        if self.cached_response is not None:
            return
        prompt = self.prompt()
        model = context['model']
        if model is None:
            raise ValueError("Model not found in context.")

        mode = context.get('mode', ModelProcessMode.TEMPERATURE_ZERO_BEAM5)
        response = model.process(prompt, mode)
        self.cached_response = response

    def predicted_image(self) -> np.array:
        if self.cached_response is None:
            raise ValueError("No response cached. Call execute() first.")
        image0 = image_from_string(self.cached_response)
        pair_index = self.task.count_examples + self.test_index
        image1 = self.task_mutator.reverse_transformation(image0, pair_index)
        return image1

    def name(self) -> str:
        return self.task_mutator.name()
