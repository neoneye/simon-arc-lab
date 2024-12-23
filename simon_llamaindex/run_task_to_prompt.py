from datetime import datetime
import sys
import os
import json
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_string_representation import *
from simon_arc_lab.image_sparse_representation import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.task_color_profile import *
from simon_arc_lab.bsp_tree import *
from simon_arc_lab.image_to_python import *
from simon_arc_lab.rle.serialize import serialize
from simon_arc_model.arc_bad_prediction import *

max_number_of_bad_predictions_per_task = 3

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data')),
    # ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

task_ids_of_interest = [
    '6fa7a44f',
    # '0b17323b',
    # '08573cc6',
    # '6c434453',
    # '21f83797',
    # '13713586',
    # '1c02dbbe',
    # '29700607',
]

arc_bad_prediction_file = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
arc_bad_prediction_dataset = None
if False:
    print(f"Loading arc-bad-prediction dataset from '{arc_bad_prediction_file}'")
    arc_bad_prediction_dataset = ARCBadPredictionDataset.load(arc_bad_prediction_file)
    # arc_bad_prediction_dataset.display_sample_records()

def format_color(colorid: int) -> str:
    colorname = IMAGETOSTRING_COLORNAME.convert_pixel_to_symbol(colorid)
    emoji = IMAGETOSTRING_EMOJI_CIRCLE_V1.convert_pixel_to_symbol(colorid)
    return f"({colorid} {colorname} {emoji})"

def format_color_list(colorid_list: list[int]) -> str:
    items = []
    for colorid in colorid_list:
        items.append(format_color(colorid))
    return ", ".join(items)

def serialize_image(image: np.array) -> str:
    histogram = Histogram.create_with_image(image)
    height, width = image.shape
    items = []
    items.append("")
    items.append(f"- width: {width}")
    items.append(f"- height: {height}")
    items.append(f"- number_of_unique_colors: {histogram.number_of_unique_colors()}")
    most_popular_colors = histogram.most_popular_color_list()
    most_popular_color_items_str = format_color_list(most_popular_colors)
    if len(most_popular_colors) == 1:
        items.append(f"- most popular color: {most_popular_color_items_str}")
    else:
        items.append(f"- most popular colors: {most_popular_color_items_str} They have the same count.")
    least_popular_colors = histogram.least_popular_color_list()
    least_popular_color_items_str = format_color_list(least_popular_colors)
    if len(least_popular_colors) == 1:
        items.append(f"- least popular color: {least_popular_color_items_str}")
    else:
        items.append(f"- least popular colors: {least_popular_color_items_str} They have the same count.")
    for color in histogram.unique_colors():
        count = histogram.get_count_for_color(color)
        color_str = format_color(color)
        items.append(f"- color {color_str}, count {count}")

    if True:
        items.append("")
        items.append("")
        items.append("representation: json")
        items.append("```json")
        json_str = json.dumps(image.tolist(), separators=(',', ':'))
        items.append(json_str)
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: digit")
        items.append("```")
        items.append(image_to_string(image))
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: color")
        items.append("```")
        items.append(image_to_string_colorname(image))
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: Excel")
        items.append("```")
        items.append(image_to_string_spreadsheet_v1(image))
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: emoji")
        items.append("```")
        items.append(image_to_string_emoji_circles_v1(image))
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: python dictionary")
        items.append("```python")
        dict_str = image_to_dictionary(image, include_size=True, background_color=histogram.most_popular_color())
        items.append(dict_str)
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: RLE")
        items.append("```")
        rle = serialize(image)
        items.append(rle)
        items.append("```")

    if True:
        items.append("")
        items.append("")
        items.append("representation: RLE transposed")
        items.append("```")
        rle = serialize(image.transpose())
        items.append(rle)
        items.append("```")

    if False:
        items.append("")
        items.append("")
        items.append("representation: BSP tree")
        items.append("```")
        bsp_node = create_bsp_tree(image, max_depth=10, verbose=False)
        bsp_str = bsp_node.tree_to_string("|")
        items.append(bsp_str)
        items.append("```")

    if True:
        config = ImageToPythonConfig()
        config.max_depth=30
        config.verbose=False
        image_to_python = ImageToPython(image, config)
        image_to_python.build()
        items.append("")
        items.append("")
        items.append("representation: numpy array")
        items.append("```python")
        items.append(image_to_python.python_code)
        items.append("```")

    return "\n".join(items)

def create_prompt_type_long(task: Task, test_index: int) -> str:
    tcp = TaskColorProfile(task)
    color_profile_prediction = tcp.predict_output_colors_for_test_index(test_index)

    items = []
    items.append("### Preamble")
    items.append("You are an expert in ARC (Abstraction & Reasoning Corpus).")
    items.append("")
    items.append("You are also an LLM so tokenization may impact how you see the image.") 
    items.append("for example three zeros may be just a single token, or it may be multiple tokens.")
    items.append("for example the color 'blue' may be just a single token, or it may be multiple tokens.")
    items.append("You cannot assume that the image you perceive is correct due to the tokenization.")
    items.append("For this reason multiple representations are provided, so you can better reason about the puzzle.")
    items.append("")
    items.append("Be aware about of keeping the number of response tokens fairly short, to minimize energy consumption.")
    items.append("In your response, please don't repeat one of the input images, since it's already in the prompt. It's a waste of tokens.")
    items.append("If you need to refer to a particular image, then write what transformation N input/output -> representation name.")
    items.append("")
    items.append("")

    rle_algorithm = """
Regarding "representation RLE".
This algorithm applies run-length encoding (RLE) to images, reducing sequences of identical pixels to shorter symbolic forms. 
It starts by noting the image's width and height, then processes each line. If an entire line is the same color, it writes that color once. 
Otherwise, it breaks the line into consecutive runs of identical pixels, using letters `a` through `z` to represent runs of lengths 2 to 27, 
followed by the color. Single pixels are just written as their color value. When a line repeats the previous one, it's marked with a comma only. 
This yields a concise, text-based compression.
"""
    items.append(rle_algorithm)
    items.append("")
    items.append("")
    items.append("Use a likert scale for your confidence level.")
    items.append("Strongly negative, weakly negative, medium, weakly positive, strongly positive.")
    items.append("Whenever there is an observation you are making, please write it down with your confidence level about it.")
    items.append("Be mindful of overstating confidence, especially in complex tasks like ARC, where multiple transformations might coexist.")
    items.append("")
    items.append("")
    items.append("- Observe first, assume cautiously, generalize carefully.")
    items.append("- Cross-check rigorously and revise judgments as needed.")
    items.append("- Treat confidence as a flexible tool, not a static declaration.")
    items.append("")
    items.append("")
    items.append("Check for Overfitting")
    items.append("Consider whether the observed pattern is just a coincidence or if it can be explained by a simpler rule.") 
    items.append("For example, if the transformations are consistent with adding red pixels at symmetrical locations or at") 
    items.append("positions defined by the coordinates of the blue pixels. Or some other color or pattern.")
    items.append("")
    items.append("")
    items.append("ARC color mapping (color id, color name, emoji)")
    for i in range(10):
        items.append(f"- {format_color(i)}")
    items.append("")
    items.append("")


    items.append("# The ARC puzzle")
    items.append("")
    for example_index in range(task.count_examples):
        if example_index > 0:
            items.append("")
            items.append("")
        input_image = task.example_input(example_index)
        output_image = task.example_output(example_index)
        items.append(f'## Train Pair {example_index} Input')
        items.append(serialize_image(input_image))
        items.append("")
        items.append("")
        items.append(f'## Train Pair {example_index} Output')
        items.append(serialize_image(output_image))

    items.append("")
    items.append("")
    input_image = task.test_input(test_index)
    # items.append(f'## Test Pair {task.count_examples} Input')
    items.append('## Test Pair Input')
    items.append(serialize_image(input_image))
    items.append("")
    items.append("")
    items.append('## Test Pair Output')
    # items.append(f'## Pair {task.count_examples} Output')
    items.append("")
    items.append("This is what the model should predict. Before giving the answer, please reflect on what transformation happens.")
    items.append("")
    items.append("My own thoughts about the output:")
    for y, (certain, color_set) in enumerate(color_profile_prediction.certain_colorset_list):
        color_list = list(color_set)
        color_list.sort()
        color_list_str = format_color_list(color_list)
        items.append(f"- I guess these colors are likely present in the output: {color_list_str}")

    if task.has_same_input_output_size_for_all_examples():
        items.append("- The output size is the same as the input size.")
    else:
        items.append("- The output size may differ from the input size.")

    items.append("")
    items.append("# Task A - Compare the training pair 0 and pair 1")
    items.append("")
    items.append("Establish reasoning steps linking inputs to their outputs.")
    items.append("")
    items.append("# Task B - Apply rules to the training pair 0")
    items.append("")
    items.append("Apply the hypothesized rule to the 'train 0 input' and see if it yields the 'train 0 output' image.")
    items.append("")
    items.append("# Task C - Apply rules to the training pair 1")
    items.append("")
    items.append("Apply the hypothesized rule to the 'train 1 input' and see if it yields the 'train 1 output' image.")
    items.append("")
    if task.count_examples > 2:
        items.append("# Task D - For all the training pairs, what transformation happens from the input to the output?")
        items.append("")
        items.append("Apply the hypothesized rules to all the remaining training pairs and see if it yields the corresponding output images.")
        items.append("")
    items.append("# Task E - Predict the output using the generalized transformation")
    items.append("")
    # items.append("Use emoji representation for the output.")
    # items.append("Use excel speadsheet representation for the output.")
    items.append("For the output, use json representation wrapped in triple backticks. Use newline to separate rows, so it's human readable.")
    items.append("")
    items.append("Rate how confident you are in your prediction.")
    items.append("")
    items.append("# Task F - double check your own answer")
    items.append("")
    items.append("Describe the transformation from the input to the output.")
    items.append("Verify that the pixels in the output are placed at the same positions that you had in mind.")
    items.append("After the double check, rate how confident you now are in your prediction.")
    items.append("")
    # items.append("# Task G")
    # items.append("create a RLE representation of the predicted output")
    # items.append("")

    result = "\n".join(items)
    # print(f"bytes: {len(result)}")
    return result

def create_prompt_type_short_json(task: Task, test_index: int) -> str:
    new_task = Task()
    for example_index in range(task.count_examples):
        input_image = task.example_input(example_index)
        output_image = task.example_output(example_index)
        new_task.append_pair(input_image, output_image, True)
    
    test_input_image = task.test_input(test_index)
    new_task.append_pair(test_input_image, None, False)

    test_output_image = task.test_output(test_index)
    # print(f"test_output_image: {test_output_image.tolist()}")

    json_string = new_task.to_arcagi1_json(True)

    items = []
    # items.append("# Requirements")
    # items.append("")
    # items.append("- The response must be markdown format. Don't use no formatting. Don't use reStructuredText.")
    # items.append("- The headings Task A, Task B, Task C must be short. Don't change the headings.")
    items.append("")
    items.append("# ARC Puzzle")
    items.append("")
    items.append("```json")
    items.append(json_string)
    items.append("```")
    items.append("")
    items.append("The `test.output` is not provided, and it's up to you to solve it.")
    items.append("")
    items.append("# Task A")
    items.append("")
    items.append("Think step by step.")
    items.append("- What does all the inputs have in common.")
    items.append("- What does all the outputs have in common.")
    items.append("- What does input/outputs have in common.")
    items.append("- Identify what transformation is happening from input to output.")
    items.append("- Verify that the transformation rules work on all input/output pairs.")
    items.append("")
    items.append("# Task B")
    items.append("")
    items.append("Go ahead and solve this ARC puzzle.")
    items.append("")
    items.append("# Task C")
    items.append("")
    items.append("Convert the `test.output` data to json format.")
    items.append("")
    items.append("The section must contain only the json data wrapped in three back quotes, like this:")
    items.append("```json")
    items.append("[[1,2,3],[4,5,6]]")
    items.append("```")
    items.append("")
    
    result = "\n".join(items)
    # print(result)
    # print(f"bytes: {len(result)}")
    return result

def create_prompt_type_short_o3_format(task: Task, test_index: int, previous_prediction: Optional[np.array]) -> str:
    items = []
    items.append('Find the common rule that maps an input grid to an output grid, given the examples below.')
    items.append('')
    for example_index in range(task.count_examples):
        if example_index > 0:
            items.append('')

        items.append(f"Example {example_index+1}:")
        items.append('')

        input_image = task.example_input(example_index)
        items.append('Input:')
        items.append(image_to_string_spaces(input_image))

        output_image = task.example_output(example_index)
        items.append('Output:')
        items.append(image_to_string_spaces(output_image))

    items.append('')
    items.append("Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself.")
    items.append('')
    items.append('Input:')
    test_input_image = task.test_input(test_index)
    items.append(image_to_string_spaces(test_input_image))
    if previous_prediction is not None:
        items.append('Maybe output:')
        items.append(image_to_string_spaces(previous_prediction))
    items.append('')
    
    result = "\n".join(items)
    return result

def create_prompt_type_short_o3_format_with_tweaks_v2(task: Task, test_index: int, previous_prediction: Optional[np.array]) -> str:
    items = []
    items.append('Find the common rule that maps an input grid to an output grid, given the examples below. Use max 100 words.')
    items.append('')
    for example_index in range(task.count_examples):
        if example_index > 0:
            items.append('')

        items.append(f"Example {example_index+1}:")
        items.append('')

        input_image = task.example_input(example_index)
        items.append('Input:')
        items.append(image_to_string_spaces(input_image))

        output_image = task.example_output(example_index)
        items.append('Output:')
        items.append(image_to_string_spaces(output_image))

    items.append('')
    items.append("Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself.")
    items.append('')
    items.append('Input:')
    test_input_image = task.test_input(test_index)
    items.append(image_to_string_spaces(test_input_image))
    if previous_prediction is not None:
        items.append('Maybe output:')
        items.append(image_to_string_spaces(previous_prediction))
    items.append('')
    
    result = "\n".join(items)
    return result


save_dir_toplevel = f'run_tasks_result/{run_id}/'
os.makedirs(save_dir_toplevel, exist_ok=True)

# create a jsonl file
jsonl_filename = f'{save_dir_toplevel}/task_to_prompt.jsonl'
print(f"Results will be saved to '{jsonl_filename}'")

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'.")

    taskset = TaskSet.load_directory(path_to_task_dir)
    # taskset.keep_tasks_with_id(set(task_ids_of_interest), verbose=False)

    if arc_bad_prediction_dataset is not None:
        # Keep only that are present in the arc-bad-prediction dataset. Remove the rest.
        task_ids_to_ignore = set()
        for task in taskset.tasks:
            task_id = task.metadata_task_id
            if arc_bad_prediction_dataset.has_records_for_task(dataset_id, task_id):
                continue
            task_ids_to_ignore.add(task_id)
        taskset.remove_tasks_by_id(task_ids_to_ignore, verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    pbar = tqdm(taskset.tasks, desc=f"Processing tasks in {groupname}", dynamic_ncols=True, leave=False)
    for task in pbar:
        task_id = task.metadata_task_id
        pbar.set_postfix_str(f"Task: {task_id}")

        def generate_prompt_for_task(task: Task, test_index: int, previous_prediction: Optional[np.array]):
            # prompt = create_prompt_type_long(task, test_index)
            # prompt = create_prompt_type_short_json(task, test_index)
            # prompt = create_prompt_type_short_o3_format(task, test_index, previous_prediction)
            prompt = create_prompt_type_short_o3_format_with_tweaks_v2(task, test_index, previous_prediction)
            # print(prompt)

            test_input = task.test_input(test_index).tolist()
            test_input_json_str = json.dumps(test_input, separators=(',', ':'))

            test_output = task.test_output(test_index).tolist()
            test_output_json_str = json.dumps(test_output, separators=(',', ':'))

            # append json to jsonl file
            jsonl_item = {
                "groupname": groupname,
                "dataset": dataset_id,
                "task": task_id,
                "test_index": test_index,
                "prompt": prompt,
                "test_input": test_input_json_str,
                "test_output": test_output_json_str,
            }
            with open(jsonl_filename, 'a') as f:
                json_str = json.dumps(jsonl_item, separators=(',', ':'))
                f.write(json_str)
                f.write("\n")

        if arc_bad_prediction_dataset is not None:
            for test_index in range(task.count_tests):
                record_list = arc_bad_prediction_dataset.find_records_for_task(dataset_id, task_id, test_index)

                # truncate to the first N bad predictions
                record_list = record_list[:max_number_of_bad_predictions_per_task]

                for record in record_list:
                    generate_prompt_for_task(task, test_index, record.predicted_output)

        for test_index in range(task.count_tests):
            generate_prompt_for_task(task, test_index, None)
