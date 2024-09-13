import json

MAX_FILE_COUNT = 100

FILENAME_PREFIX = 'symmetry_'

# File paths
input_file = 'simon_arc_dataset_run/dataset_solve_symmetry.jsonl'
output_folder = 'simon_arc_dataset_run/output_folder/'

# Read the JSONL file and export each row as a separate JSON file
with open(input_file, 'r') as f:
    for index, line in enumerate(f, start=1):
        if index > MAX_FILE_COUNT:
            print(f"Reached the maximum file count of {MAX_FILE_COUNT}. Stopping the export.")
            break

        # Load the current line as a JSON object
        json_data = json.loads(line)
        
        # Define the output file name based on the row number
        output_file = f"{output_folder}{FILENAME_PREFIX}{index}.json"
        
        # Write the JSON object to a new file
        with open(output_file, 'w') as outfile:
            json.dump(json_data, outfile, separators=(',', ':'))

        print(f"Exported: {output_file}")
