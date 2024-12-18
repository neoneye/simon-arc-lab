from datetime import datetime
import sys
import os
import json
from tqdm import tqdm
import numpy as np
from dotenv import dotenv_values
from llama_index.llms.openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_string_representation import *
from simon_arc_lab.image_sparse_representation import *
from simon_arc_lab.histogram import Histogram
from simon_arc_lab.task_color_profile import *
from simon_arc_lab.bsp_tree import *
from simon_arc_lab.rle.serialize import serialize

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

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
    '0b17323b',
    '08573cc6',
    '6c434453',
    '21f83797',
]

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

    if True:
        items.append("")
        items.append("")
        items.append("representation: BSP tree")
        items.append("```")
        bsp_node = create_bsp_tree(image, max_depth=10, verbose=False)
        bsp_str = bsp_node.tree_to_string("|")
        items.append(bsp_str)
        items.append("```")

    return "\n".join(items)

def create_prompt_for_task(task: Task, test_index: int) -> str:
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

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.keep_tasks_with_id(set(task_ids_of_interest), verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    pbar = tqdm(taskset.tasks, desc=f"Processing tasks in {groupname}", dynamic_ncols=True)
    for task in pbar:
        task_id = task.metadata_task_id
        pbar.set_postfix_str(f"Task: {task_id}")

        for test_index in range(task.count_tests):
            prompt = create_prompt_for_task(task, test_index)
            filename = f'{task_id}_test{test_index}_prompt.md'
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'w') as f:
                f.write(prompt)

exit()

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=dotenv_dict['OPENAI_API_KEY'],
)

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
