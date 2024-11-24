import torch
from torch.utils.data import Dataset
import json
import os
import random
from typing import Optional

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        src = torch.tensor(self.xs[idx], dtype=torch.long)
        ys = torch.tensor(self.ys[idx], dtype=torch.long)
        return src, ys

    @classmethod
    def load_jsonl_files(cls, path_to_dir: str, truncate: Optional[int]) -> 'MyDataset':
        """
        Create a MyDataset object from a directory containing JSONL files.
        """
        jsonl_file_paths = []
        for subdir, dirs, files in os.walk(path_to_dir):
            for file in sorted(files):
                # good = file.startswith('28e73c20') or file.startswith('1f876c06') or file.startswith('2281f1f4')
                # if not good:
                #     continue
                if file.endswith('.jsonl'):
                    file_path = os.path.join(subdir, file)
                    jsonl_file_paths.append(file_path)
        print('MyDataset.load_jsonl_files() jsonl_file_paths:', len(jsonl_file_paths))
        if truncate is not None:
            jsonl_file_paths = jsonl_file_paths[:truncate]
            print('MyDataset.load_jsonl_files() truncated jsonl_file_paths:', len(jsonl_file_paths), 'paths:', jsonl_file_paths)

        all_xs = []
        all_ys = []
        for input_file in jsonl_file_paths:
            pending_xs = []
            pending_ys = []
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    xs = data['xs']
                    ys = data['ys']
                    pending_xs.append(xs)
                    pending_ys.append(ys)
            if False:
                # truncate to half
                pending_xs = pending_xs[:len(pending_xs)//2]
                pending_ys = pending_ys[:len(pending_ys)//2]
            all_xs.extend(pending_xs)
            all_ys.extend(pending_ys)

        # shuffle the data
        indices = list(range(len(all_xs)))
        random.Random(123).shuffle(indices)
        all_xs = [all_xs[i] for i in indices]
        all_ys = [all_ys[i] for i in indices]

        print('MyDataset.load_jsonl_files() all_xs:', len(all_xs))
        return cls(all_xs, all_ys)
