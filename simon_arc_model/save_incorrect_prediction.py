import json

def save_incorrect_prediction(
    jsonl_path: str,
    dataset_id: str,
    task_id: str,
    test_index: int,
    predict: list[list[int]],
    metadata: str
) -> None:
    """
    Append the incorrect prediction to the JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.
        dataset_id (str): Identifier for the dataset, eg. 'ARC-AGI', 'ConceptARC'.
        task_id (str): Identifier for the task, eg. '1d398264', 'fcc82909'.
        test_index (int): Index of the test case.
        predict (list[list[int]]): The incorrect predicted output image.
        metadata (str): Additional metadata information.

    Returns:
        None
    """
    record = {
        "dataset": dataset_id,
        "task": task_id,
        "test_index": test_index,
        "predict": predict,
        "metadata": metadata
    }
    s = json.dumps(record, separators=(',', ':')) + '\n'
    with open(jsonl_path, 'a') as f:
        f.write(s)
