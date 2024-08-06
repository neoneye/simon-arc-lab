import random
import json
import os
from tqdm import tqdm
from .plot import *

class DatasetGenerator:
    def __init__(self, dataset_names: list[str], benchmark_dataset_name: str, generate_dataset_item_list_fn):
        self.dataset_names = dataset_names
        self.benchmark_dataset_name = benchmark_dataset_name
        self.generate_dataset_item_list_fn = generate_dataset_item_list_fn

        self.generated_dataset_items = None

    def generate(self, seed: int, max_num_samples=1000, max_byte_size=1024*1024):
        dataset = []
        dataset_byte_size = 0
        stop = False
        with tqdm(total=max_num_samples, desc="Generating dataset", ncols=100, unit="sample", mininterval=1.0, dynamic_ncols=True, leave=True) as pbar:
            for i in range(max_num_samples):
                if stop:
                    break
                items = self.generate_dataset_item_list_fn(seed + i + 1000)
                for item in items:
                    bytes = len(json.dumps(item))
                    if dataset_byte_size + bytes > max_byte_size:
                        stop = True
                        break
                    if len(dataset) >= max_num_samples:
                        stop = True
                        break
                    dataset_byte_size += bytes
                    dataset.append(item)
                    pbar.update(1)
        random.Random(seed).shuffle(dataset)

        self.generated_dataset_items = dataset

    def save(self, filename: str):
        dataset_items = self.generated_dataset_items
        with open(filename, 'w') as f:
            for item in dataset_items:
                f.write(json.dumps(item) + '\n')
        file_size = os.path.getsize(filename)
        print(f"Generated {len(dataset_items)} samples, saved to {filename}, file size: {file_size} bytes.")

    def inspect(self):
        plot_prompt_length_distribution(self.generated_dataset_items)
        plot_response_length_distribution(self.generated_dataset_items)
