import json
from model.runner import *

# Load dataset
with open('dataset_solve_translate.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model122'

# Initialize runner
runner = Runner(model_directory, 512)
runner.run(dataset)
