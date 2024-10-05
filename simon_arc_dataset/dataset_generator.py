# IDEA: Reject dataset items that are too big to fit inside the context length limit, 1024 tokens.
#
# IDEA: Retry mechanism, if the model fails to generate a response, then retry with a different seed.
import random
import json
import os
from tqdm import tqdm
from .plot import *

class DatasetGenerator:
    def __init__(self, generate_dataset_item_list_fn):
        self.generate_dataset_item_list_fn = generate_dataset_item_list_fn
        self.row_strings = None
        self.dataset_items = None

    def generate(self, seed: int, max_num_samples=1000, max_byte_size=1024*1024):
        row_strings = []
        dataset_items = []
        file_size = 0
        stop = False
        with tqdm(total=max_num_samples, desc="Generating dataset", ncols=100, unit="sample", mininterval=1.0, dynamic_ncols=True, leave=True) as pbar:
            for i in range(max_num_samples):
                if stop:
                    break
                items = self.generate_dataset_item_list_fn(seed + i + 1000)
                for item in items:
                    row_string = json.dumps(item, separators=(',', ':')) + '\n'
                    bytes = len(row_string)
                    if file_size + bytes > max_byte_size:
                        stop = True
                        break
                    if len(row_strings) >= max_num_samples:
                        stop = True
                        break
                    file_size += bytes
                    row_strings.append(row_string)
                    dataset_items.append(item)
                    pbar.update(1)
        random.Random(seed).shuffle(row_strings)

        self.row_strings = row_strings
        self.dataset_items = dataset_items

    def save(self, file_path: str):
        """
        Save the generated dataset to a "jsonl" file
        """
        row_strings = self.row_strings
        with open(file_path, 'w') as f:
            for row_string in row_strings:
                f.write(row_string)
        file_size = os.path.getsize(file_path)
        print(f"Generated {len(row_strings)} samples, saved to {file_path}, file size: {file_size} bytes.")

    def inspect(self):
        plot_prompt_length_distribution(self.dataset_items)
        plot_response_length_distribution(self.dataset_items)
