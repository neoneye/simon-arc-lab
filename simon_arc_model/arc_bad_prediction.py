"""
Wrapper around the "ARC-Bad-Prediction" dataset.
https://huggingface.co/datasets/neoneye/arc-bad-prediction
"""
import json
from dataclasses import dataclass
import numpy as np

@dataclass
class ARCBadPredictionRecord:
    """
    A single record of the "ARC-Bad-Prediction" dataset.
    """
    dataset: str
    task: str
    test_index: int
    predicted_output: np.array # NumPy array with dtype=np.uint8
    metadata: str
    line_number: int

class ARCBadPredictionDataset:
    """
    Class to handle loading and interacting with the "ARC-Bad-Prediction" dataset.
    """

    def __init__(self, records: list[ARCBadPredictionRecord]) -> None:
        self.records = records

        # Create a set of unique (dataset_id, task_id) pairs
        dataset_task = set()
        for record in records:
            dataset_id = record.dataset
            task_id = record.task
            dataset_task.add((dataset_id, task_id))

        self.dataset_task = dataset_task


    @classmethod
    def load(cls, file_path: str) -> 'ARCBadPredictionDataset':
        """
        Loads the "ARC-Bad-Prediction" dataset from a JSON Lines file.

        Parameters:
            file_path (str): The path to the JSON Lines (.jsonl) file containing the dataset.
        """
        records = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    record_json: dict = json.loads(line)

                    # Extract and validate required fields
                    dataset: str = record_json['dataset']
                    task: str = record_json['task']
                    test_index: int = record_json['test_index']
                    predicted_output_list: list[list[int]] = record_json['predicted_output']
                    metadata: str = record_json['metadata']

                    # Convert 'predicted_output' to a NumPy array
                    predicted_output: np.array = np.array(predicted_output_list, dtype=np.uint8)

                    # Create a record instance
                    record: ARCBadPredictionRecord = ARCBadPredictionRecord(
                        dataset=dataset,
                        task=task,
                        test_index=test_index,
                        predicted_output=predicted_output,
                        metadata=metadata,
                        line_number=line_number
                    )

                    records.append(record)

                except json.JSONDecodeError as e:
                    print(f"JSON decoding error on line {line_number}: {e}")
                except KeyError as e:
                    print(f"Missing key {e} on line {line_number}")
                except TypeError as e:
                    print(f"Type error on line {line_number}: {e}")
                except Exception as e:
                    print(f"Unexpected error on line {line_number}: {e}")
        return ARCBadPredictionDataset(records)

    def display_sample_records(self, sample_size: int = 5) -> None:
        """
        Displays basic information and samples from the loaded dataset.

        Parameters:
            sample_size (int): Number of sample records to display.
        """
        total_records: int = len(self.records)
        print("Dataset Loaded Successfully!")
        print(f"Total Records: {total_records}\n")

        if total_records == 0:
            print("No records to display.")
            return
        print(f"Number of unique tasks in the arc-bad-prediction dataset: {len(self.dataset_task)}")

        sample_size = min(sample_size, total_records)
        print(f"Displaying first {sample_size} records:\n")

        for i in range(sample_size):
            record: ARCBadPredictionRecord = self.records[i]
            print(f"Record {i + 1}:")
            print(f"  Dataset: {record.dataset}")
            print(f"  Task: {record.task}")
            print(f"  Test Index: {record.test_index}")
            print(f"  Predicted Output:\n{record.predicted_output}")
            print(f"  Metadata: {record.metadata}")
            print(f"  Line Number: {record.line_number}\n")

    def has_records_for_task(self, dataset_id: str, task_id: str) -> bool:
        """
        Fast way to check if there are records for a given (dataset_id, task_id) pair.

        Returns True if there are 1 or more records for a given (dataset_id, task_id) pair.
        """
        key = (dataset_id, task_id)
        return key in self.dataset_task

    def find_records_for_task(self, dataset_id: str, task_id: str, task_count_tests: int) -> list[ARCBadPredictionRecord]:
        """
        Slow way to find records for a given (dataset_id, task_id) pair.

        Returns a list of records for a given (dataset_id, task_id) pair.
        """
        find_key = (dataset_id, task_id)
        found_records = []
        for record in self.records:
            record_dataset_id = record.dataset
            record_task_id = record.task
            record_key = (record_dataset_id, record_task_id)
            if find_key != record_key:
                continue

            test_index = record.test_index
            if test_index >= task_count_tests:
                print(f"Skipping task: {task_id}, due to test index {test_index} is out of range.")
                continue
            found_records.append(record)

        return found_records

if __name__ == "__main__":
    dataset_file: str = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
    arc_bad_prediction_dataset = ARCBadPredictionDataset.load(dataset_file)
    arc_bad_prediction_dataset.display_sample_records()
