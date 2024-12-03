"""
Populate the "ARC-Bad-Prediction" dataset.
https://huggingface.co/datasets/neoneye/arc-bad-prediction

When encountering an incorrect prediction, that is somewhat ok, 
then it may be a candidate for the "ARC-Bad-Prediction" dataset.
"""
import json
import os
import numpy as np
from typing import Optional
from .work_item import WorkItem
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.image_string_representation import image_to_string

class TrackIncorrectPrediction:
    def __init__(self, jsonl_path: str, taskid_testindex_predictedoutput_set: set):
        self.jsonl_path = jsonl_path
        self.taskid_testindex_predictedoutput_set = taskid_testindex_predictedoutput_set

    @classmethod
    def load_from_jsonl(cls, jsonl_path: str) -> 'TrackIncorrectPrediction':
        """
        Load the incorrect predictions from the JSONL file.

        Mechanism to prevent duplicate entries in the jsonl file.
        if it's already in the file, then there is no need to add it to the file again.
        """
        # check if the file exist, if not then return an empty set.
        if not os.path.isfile(jsonl_path):
            print(f"TrackIncorrectPrediction. File does not exist: {jsonl_path}")
            return TrackIncorrectPrediction(jsonl_path, set())
        
        # Populate the set with the existing entries in the jsonl file.
        taskid_testindex_predictedoutput_set = set()
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                task_id = record['task']
                test_index = record['test_index']
                predicted_output = record['predicted_output']

                # Inside the jsonl file, the "predicted_output" is a list[list[int]].
                predicted_output_nparray = np.array(predicted_output)
                predicted_output_key = image_to_string(predicted_output_nparray)
                taskid_testindex_predictedoutput_set.add((task_id, test_index, predicted_output_key))
        print(f"TrackIncorrectPrediction. Loaded {len(taskid_testindex_predictedoutput_set)} incorrect predictions from: {jsonl_path}")
        return TrackIncorrectPrediction(jsonl_path, taskid_testindex_predictedoutput_set)

    @classmethod
    def save_incorrect_prediction(cls, jsonl_path: str, dataset_id: str, task_id: str, test_index: int, predicted_output: np.array, metadata: str) -> None:
        """
        Append the incorrect prediction to the JSONL file.

        Args:
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

    def track_incorrect_prediction(self, work_item: WorkItem, dataset_id: str, predicted_output: Optional[np.array], metadata: str) -> None:
        """
        Track incorrect predictions and save them to a JSONL file.

        Do some filtering, so that only interesting incorrect predictions are saved:
        - Skip predictions that are correct, since there is nothing wrong.
        - Skip predictions that are identical to the input image, since this is the starting point anyways.
        - Skip predictions that have less than 2 unique colors, since they are not interesting.
        """
        task = work_item.task
        test_index = work_item.test_index
        task_id = task.metadata_task_id
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
        predicted_output_key = image_to_string(predicted_output)
        if (task_id, test_index, predicted_output_key) in self.taskid_testindex_predictedoutput_set:
            print(f"Skipping duplicate incorrect prediction. {task_id} {test_index}")
            return
        print(f"Added prediction. {task_id} {test_index}")
        self.save_incorrect_prediction(
            self.jsonl_path,
            dataset_id,
            task_id,
            test_index,
            predicted_output,
            metadata,
        )
        self.taskid_testindex_predictedoutput_set.add((task_id, test_index, predicted_output_key))
