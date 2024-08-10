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
