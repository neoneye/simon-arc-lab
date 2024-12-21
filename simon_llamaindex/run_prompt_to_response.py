import os
import json
import time
from datetime import datetime
import sys
from typing import Optional
import numpy as np
from tqdm import tqdm
from dotenv import dotenv_values
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.json_from_response import json_from_response
from simon_arc_lab.image_from_json_array import image_from_json_array
from simon_arc_lab.show_prediction_result import show_prediction_result

class TaskToPromptItem:
    def __init__(self, json_dict: dict, row_index: int, groupname: str, dataset_id: str, task_id: str, test_index: int, prompt: str, test_input: np.array, test_output: np.array):
        if not isinstance(groupname, str):
            raise ValueError(f"Expected groupname to be a string, but got: {groupname}")
        if not isinstance(dataset_id, str):
            raise ValueError(f"Expected dataset_id to be a string, but got: {dataset_id}")
        if not isinstance(task_id, str):
            raise ValueError(f"Expected task_id to be a string, but got: {task_id}")
        if not isinstance(test_index, int):
            raise ValueError(f"Expected test_index to be an int, but got: {test_index}")
        if not isinstance(prompt, str):
            raise ValueError(f"Expected prompt to be a string, but got: {prompt}")
        if not isinstance(test_input, np.ndarray):
            raise ValueError(f"Expected test_output to be a np.ndarray, but got: {test_input}")
        if not isinstance(test_output, np.ndarray):
            raise ValueError(f"Expected test_output to be a np.ndarray, but got: {test_output}")

        self.json_dict = json_dict
        self.row_index = row_index
        self.groupname = groupname
        self.dataset_id = dataset_id
        self.task_id = task_id
        self.test_index = test_index
        self.prompt = prompt
        self.test_input = test_input
        self.test_output = test_output

    @staticmethod
    def from_json_string(json_string: str, row_index: int) -> 'TaskToPromptItem':
        if not isinstance(json_string, str):
            raise ValueError(f"Expected json_string to be a string, but got: {json_string}")
        if not isinstance(row_index, int):
            raise ValueError(f"Expected row_index to be an int, but got: {row_index}")
        
        dict = json.loads(json_string)

        # test_input
        test_input_json = dict['test_input']
        test_input_list_of_list_of_int = json.loads(test_input_json)
        test_input = np.array(test_input_list_of_list_of_int, dtype=np.uint8)

        # test_output
        test_output_json = dict['test_output']
        test_output_list_of_list_of_int = json.loads(test_output_json)
        test_output = np.array(test_output_list_of_list_of_int, dtype=np.uint8)
        return TaskToPromptItem(
            json_dict=dict,
            row_index=row_index,
            groupname=dict['groupname'],
            dataset_id=dict['dataset'],
            task_id=dict['task'],
            test_index=dict['test_index'],
            prompt=dict['prompt'],
            test_input=test_input,
            test_output=test_output
        )

    @staticmethod
    def load_json_file(jsonl_path: str, show: bool, truncate: Optional[int] = None) -> list['TaskToPromptItem']:
        item_list = []
        with open(jsonl_path, 'r') as f:
            json_rows = f.readlines()
            for row_index, json_string in enumerate(json_rows):
                item = TaskToPromptItem.from_json_string(json_string, row_index)
                item_list.append(item)
                if truncate is not None and row_index >= truncate:
                    break
                if show:
                    print(item)
        return item_list

    def __str__(self):
        return f"TaskToPromptItem(row={self.row_index} task_id={self.task_id} test_index={self.test_index} prompt.len={len(self.prompt)})"

    def __repr__(self):
        return self.__str__()

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241221_202038/task_to_prompt.jsonl'
task_to_prompt_item_list = TaskToPromptItem.load_json_file(jsonl_file, show=True, truncate=5)

print(f"Number of task_to_prompt_item_list: {len(task_to_prompt_item_list)}")

llm = Ollama(model="llama3.1:latest", request_timeout=120.0, temperature=0.0)

save_dir = f'run_tasks_result/{run_id}/'
os.makedirs(save_dir, exist_ok=True)

pbar = tqdm(task_to_prompt_item_list, desc=f"Processing tasks", dynamic_ncols=True, leave=False)
for item in pbar:
    pbar.set_postfix_str(f"Task: {item.task_id}")

    prompt = item.prompt

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content=prompt),
    ]
    resp = llm.stream_chat(messages)

    print("stream response:")
    start_time = time.time()
    text_items = []
    for response_index, r in enumerate(resp):
        text_item = r.delta
        text_items.append(text_item)
        print(text_item, end="")
        if response_index > 10:
            error_message = "... Too long response, skipping the rest"
            print(error_message)
            text_items.append(error_message)
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n\nelapsed: {elapsed_time:.2f} seconds")
    print("\n\nfull response:")
    text = "".join(text_items)
    print(text)

    response_json = json_from_response(text)
    print(f"response_json: {response_json}")
    predicted_output_image = image_from_json_array(response_json, padding=255)
    print(f"image: {predicted_output_image.tolist()}")

    is_correct = np.array_equal(item.test_output, predicted_output_image)
    status = "correct" if is_correct else "incorrect"

    filename_items_optional = [
        item.groupname,
        item.task_id,
        f'test{item.test_index}',
        status,
    ]
    filename_items = [item for item in filename_items_optional if item is not None]
    filename = '_'.join(filename_items) + '.png'

    save_path = os.path.join(save_dir, filename)

    title = f"{item.task_id} test_index={item.test_index} {status}"
    show_prediction_result(item.test_input, predicted_output_image, item.test_output, title, show_grid=True, save_path=save_path)

    print("DONE")
    break
