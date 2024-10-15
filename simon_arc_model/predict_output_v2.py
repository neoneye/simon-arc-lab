from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *
from simon_arc_lab.task_mutator import *
from simon_arc_lab.image_distort import *
from simon_arc_lab.image_noise import *
from simon_arc_lab.rle.deserialize import *
from simon_arc_model.model import Model, ModelProcessMode
from .predict_output_base import PredictOutputBase

class PredictOutputV2(PredictOutputBase):
    def __init__(self, task: Task, test_index: int, task_mutator_class: type, previous_predicted_image: np.array = None):
        if not issubclass(task_mutator_class, TaskMutatorBase):
            raise TypeError(f"{task_mutator_class.__name__} must be a subclass of TaskMutatorBase")
        
        self.task = task
        self.test_index = test_index
        self.task_mutator = task_mutator_class(task)
        # self.task_mutator = TaskMutatorDoNothing(task)
        # self.task_mutator = TaskMutatorTranspose(task)
        # self.task_mutator = TaskMutatorTransposeSoInputIsMostCompact(task)
        self.previous_predicted_image = previous_predicted_image
        self.cached_prompt = None
        self.cached_response = None

    def prompt(self) -> str:
        if self.cached_prompt:
            return self.cached_prompt

        task_without_test_output = self.task_mutator.transformed_task()
        task_without_test_output.set_all_test_outputs_to_none()

        if self.previous_predicted_image is not None:
            image_index = self.task.count_examples + self.test_index
            task_without_test_output.output_images[image_index] = self.previous_predicted_image

        # TODO: insert the previous_predicted_image into the prompt with the transformation applied, eg. transpose/rotate

        # Provide an earlier predicted output, as part of the prompt.
        # image_index = self.task.count_examples + self.test_index
        # expected_output = self.task.output_images[image_index].copy()
        # expected_output = image_distort(expected_output, 1, 5, 42)
        # expected_output = image_noise_one_pixel(expected_output, 0)
        # if isinstance(self.task_mutator, TaskMutatorTranspose):
        #     expected_output = np.transpose(expected_output)
        # task_without_test_output.output_images[image_index] = expected_output

        task_formatter = TaskFormatterRLECompact(task_without_test_output)
        test_output_id = task_formatter.test_output_id(self.test_index)
        input = task_formatter.to_string()
        prompt = f"SIMON-SOLVE-V1, {test_output_id}, predict image\n{input}"

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
        # IDEA: Save multiple responses. Currently only saves 1 response.
        response = model.process(prompt, mode)
        self.cached_response = response

    def predicted_image(self) -> np.array:
        if self.cached_response is None:
            raise ValueError("No response cached. Call execute() first.")
        image0 = deserialize(self.cached_response)
        pair_index = self.task.count_examples + self.test_index
        image1 = self.task_mutator.reverse_transformation(image0, pair_index)
        return image1

    def name(self) -> str:
        return self.task_mutator.name()
