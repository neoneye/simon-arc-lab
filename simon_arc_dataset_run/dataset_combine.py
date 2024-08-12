import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

seed = 59
random.seed(seed)

# Define the input file paths
dataset_dir = os.path.dirname(__file__)
file_names = [
    'dataset_cellular_automaton.jsonl',
    'dataset_histogram.jsonl',
    'dataset_image.jsonl',
    'dataset_image_pair.jsonl',
    'dataset_mass.jsonl',
    'dataset_scale.jsonl',
    'dataset_shape.jsonl',
    'dataset_solve_color.jsonl',
    'dataset_solve_rotate.jsonl',
    'dataset_solve_translate.jsonl',
    'dataset_symmetry.jsonl',
]

# Number of rows in the output file
output_rows = 200000

# Calculate the number of rows to sample from each file
rows_per_file = output_rows // len(file_names)

# List to hold the sampled rows
sampled_rows = []

# Function to read and sample rows from a JSONL file
def sample_rows(file_path, num_rows):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sampled = random.sample(lines, num_rows)
        return sampled

# Sample rows from each file
for filename in file_names:
    file_path = os.path.join(dataset_dir, filename)
    sampled_rows.extend(sample_rows(file_path, rows_per_file))

# Shuffle the combined rows
random.shuffle(sampled_rows)

# Write the sampled rows to the output file
output_file_path = os.path.join(dataset_dir, 'dataset_combine.jsonl')
with open(output_file_path, 'w') as output_file:
    for row in sampled_rows:
        output_file.write(row)

print(f"Output file created: {output_file_path}")
