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
    predicted_output: np.array
    metadata: str

class ARCBadPredictionDataset:
    """
    Class to handle loading and interacting with the "ARC-Bad-Prediction" dataset.
    """

    def __init__(self) -> None:
        self.records: list[ARCBadPredictionRecord] = []

    @classmethod
    def load(cls, file_path: str) -> 'ARCBadPredictionDataset':
        """
        Loads the "ARC-Bad-Prediction" dataset from a JSON Lines file.

        Parameters:
            file_path (str): The path to the JSON Lines (.jsonl) file containing the dataset.
        """
        result_dataset = ARCBadPredictionDataset()
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
                    predicted_output: np.array = np.array(predicted_output_list)

                    # Create a record instance
                    record: ARCBadPredictionRecord = ARCBadPredictionRecord(
                        dataset=dataset,
                        task=task,
                        test_index=test_index,
                        predicted_output=predicted_output,
                        metadata=metadata
                    )

                    result_dataset.records.append(record)

                except json.JSONDecodeError as e:
                    print(f"JSON decoding error on line {line_number}: {e}")
                except KeyError as e:
                    print(f"Missing key {e} on line {line_number}")
                except TypeError as e:
                    print(f"Type error on line {line_number}: {e}")
                except Exception as e:
                    print(f"Unexpected error on line {line_number}: {e}")
        return result_dataset

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

        sample_size = min(sample_size, total_records)
        print(f"Displaying first {sample_size} records:\n")

        for i in range(sample_size):
            record: ARCBadPredictionRecord = self.records[i]
            print(f"Record {i + 1}:")
            print(f"  Dataset: {record.dataset}")
            print(f"  Task: {record.task}")
            print(f"  Test Index: {record.test_index}")
            print(f"  Predicted Output:\n{record.predicted_output}")
            print(f"  Metadata: {record.metadata}\n")

if __name__ == "__main__":
    dataset_file: str = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
    arc_bad_prediction_dataset = ARCBadPredictionDataset.load(dataset_file)
    arc_bad_prediction_dataset.display_sample_records()
