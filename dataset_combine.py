import random

seed = 42

# Define the input file paths
file_paths = [
    'dataset_solve_color.jsonl',
    'dataset_solve_rotate.jsonl',
    'dataset_solve_translate.jsonl'
]

# Number of rows in the output file
output_rows = 100000

# Calculate the number of rows to sample from each file
rows_per_file = output_rows // len(file_paths)

# List to hold the sampled rows
sampled_rows = []

# Function to read and sample rows from a JSONL file
def sample_rows(file_path, num_rows):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sampled = random.sample(lines, num_rows)
        return sampled

# Sample rows from each file
for file_path in file_paths:
    sampled_rows.extend(sample_rows(file_path, rows_per_file))

# Shuffle the combined rows
random.Random(seed).shuffle(sampled_rows)

# Write the sampled rows to the output file
output_file_path = 'dataset_combine.jsonl'
with open(output_file_path, 'w') as output_file:
    for row in sampled_rows:
        output_file.write(row)

print(f"Output file created: {output_file_path}")