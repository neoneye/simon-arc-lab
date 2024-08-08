import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_model.runner import *

path_to_dataset_jsonl = os.path.join(PROJECT_ROOT, 'dataset_solve_translate.jsonl')

# Load dataset
with open(path_to_dataset_jsonl, 'r') as f:
    dataset = [json.loads(line) for line in f]

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model168'

# Initialize runner
runner = Runner(model_directory, 512)
runner.run(dataset)
