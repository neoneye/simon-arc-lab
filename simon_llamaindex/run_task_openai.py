from datetime import datetime
import sys
import os
import numpy as np
from dotenv import dotenv_values
from llama_index.llms.openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_string_representation import *
from simon_arc_lab.histogram import Histogram

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
    '0b17323b'
]

def format_color(colorid: int) -> str:
    colorname = IMAGETOSTRING_COLORNAME.convert_pixel_to_symbol(colorid)
    emoji = IMAGETOSTRING_EMOJI_CIRCLE_V1.convert_pixel_to_symbol(colorid)
    return f"(id{colorid} {colorname} {emoji})"

def format_color_list(colorid_list: list[int]) -> str:
    items = []
    for colorid in colorid_list:
        items.append(format_color(colorid))
    return ", ".join(items)

def serialize_image(image: np.array) -> str:
    histogram = Histogram.create_with_image(image)
    height, width = image.shape
    items = []
    items.append("details:")
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
    items.append("")
    items.append("")
    items.append("representation: digits")
    items.append("```")
    items.append(image_to_string(image))
    items.append("```")
    items.append("")
    items.append("")
    items.append("representation: color names")
    items.append("```")
    items.append(image_to_string_colorname(image))
    items.append("```")
    items.append("")
    items.append("")
    items.append("representation: Excel")
    items.append("```")
    items.append(image_to_string_spreadsheet_v1(image))
    items.append("```")
    items.append("")
    items.append("")
    items.append("representation: emoji")
    items.append("```")
    items.append(image_to_string_emoji_circles_v1(image))
    items.append("```")
    return "\n".join(items)

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.keep_tasks_with_id(set(task_ids_of_interest), verbose=False)

    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Number of tasks for processing: {len(taskset.tasks)}")

    for task in taskset.tasks:
        task_id = task.metadata_task_id
        print(f"Task id: {task_id}")

        for example_index in range(task.count_examples):
            input_image = task.example_input(example_index)
            print(f"Example: {example_index}")
            print("Input:")
            s = serialize_image(input_image)
            print(s)
            print(f"bytes: {len(s)}")
            break

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
