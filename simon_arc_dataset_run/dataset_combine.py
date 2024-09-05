import os
import sys
import random
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

seed = 162
random.seed(seed)

# Define the input file paths
dataset_dir = os.path.dirname(__file__)
file_names = [
    'dataset_cellular_automaton.jsonl',
    'dataset_dilation.jsonl',
    'dataset_erosion.jsonl',
    'dataset_histogram.jsonl',
    'dataset_image.jsonl',
    'dataset_image_pair.jsonl',
    'dataset_mass.jsonl',
    'dataset_scale.jsonl',
    'dataset_shape.jsonl',
    'dataset_solve_bool.jsonl',
    'dataset_solve_boundingbox.jsonl',
    'dataset_solve_color.jsonl',
    'dataset_solve_compress.jsonl',
    'dataset_solve_cross.jsonl',
    'dataset_solve_edge.jsonl',
    'dataset_solve_erosion.jsonl',
    'dataset_solve_flip.jsonl',
    'dataset_solve_fractal.jsonl',
    'dataset_solve_gravity.jsonl',
    'dataset_solve_grid.jsonl',
    'dataset_solve_half.jsonl',
    'dataset_solve_halfplane.jsonl',
    'dataset_solve_mask.jsonl',
    'dataset_solve_mass.jsonl',
    'dataset_solve_outline.jsonl',
    'dataset_solve_probecolor.jsonl',
    'dataset_solve_ray.jsonl',
    'dataset_solve_reverse.jsonl',
    'dataset_solve_rotate.jsonl',
    'dataset_solve_scale.jsonl',
    'dataset_solve_span.jsonl',
    'dataset_solve_skew.jsonl',
    'dataset_solve_symmetry.jsonl',
    'dataset_solve_translate.jsonl',
    'dataset_solve_zindex.jsonl',
    'dataset_symmetry.jsonl',
]

# Number of rows in the output file
output_rows = 300000

# INPUT_BYTE_LIMIT = 512 - 50
INPUT_BYTE_LIMIT = 1024 - 50

# Calculate the number of rows to sample from each file
rows_per_file = output_rows // len(file_names)

# List to hold the sampled rows
sampled_rows = []

# Function to check if the input field's byte length is within the limit
def is_within_byte_limit(input_text, byte_limit=512):
    # Encode the input text to UTF-8 and check its byte length
    return len(input_text.encode('utf-8')) <= byte_limit

# Function to read and sample rows from a JSONL file
def sample_rows(file_path, num_rows):
    print(f"Sampling {num_rows} rows from {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_lines = []
        for line in lines:
            data = json.loads(line)
            if is_within_byte_limit(data.get("input", ""), byte_limit=INPUT_BYTE_LIMIT):
                valid_lines.append(line)
        percent_valid = len(valid_lines) * 100 // len(lines)
        print(f"Valid rows: {len(valid_lines)} ({percent_valid}%)")
        return random.sample(valid_lines, min(num_rows, len(valid_lines)))

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
