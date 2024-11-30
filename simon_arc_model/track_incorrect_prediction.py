import json
import numpy as np
from typing import Optional
from .work_item import WorkItem
from simon_arc_lab.histogram import Histogram

def save_incorrect_prediction(
    jsonl_path: str,
    dataset_id: str,
    task_id: str,
    test_index: int,
    predicted_output: np.array,
    metadata: str
) -> None:
    """
    Append the incorrect prediction to the JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.
        dataset_id (str): Identifier for the dataset, eg. 'ARC-AGI', 'ConceptARC'.
        task_id (str): Identifier for the task, eg. '1d398264', 'fcc82909'.
        test_index (int): Index of the test case.
        predicted_output (np.array): The incorrect predicted output image.
        metadata (str): Additional metadata information.

    Returns:
        None
    """
    record = {
        "dataset": dataset_id,
        "task": task_id,
        "test_index": test_index,
        "predicted_output": predicted_output.tolist(),
        "metadata": metadata
    }
    s = json.dumps(record, separators=(',', ':')) + '\n'
    with open(jsonl_path, 'a') as f:
        f.write(s)
        f.flush()

def track_incorrect_prediction(
    work_item: WorkItem, 
    jsonl_path: str, 
    dataset_id: str, 
    predicted_output: Optional[np.array], 
    metadata: str
) -> None:
    """
    Track incorrect predictions and save them to a JSONL file.

    Do some filtering, so that only interesting incorrect predictions are saved:
    - Skip predictions that are correct, since there is nothing wrong.
    - Skip predictions that are identical to the input image, since this is the starting point anyways.
    - Skip predictions that have less than 2 unique colors, since they are not interesting.
    """
    task = work_item.task
    test_index = work_item.test_index
    if predicted_output is None:
        return
    if np.array_equal(predicted_output, task.test_output(test_index)):
        return
    if np.array_equal(predicted_output, task.test_input(test_index)):
        return
    histogram = Histogram.create_with_image(predicted_output)
    if histogram.number_of_unique_colors() < 2:
        # print(f"Skipping incorrect prediction with less than 2 unique colors.")
        return
    save_incorrect_prediction(
        jsonl_path,
        dataset_id,
        task.metadata_task_id,
        test_index,
        predicted_output,
        metadata,
    )
