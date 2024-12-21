import os
import json
import time
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

class TaskToPromptItem:
    def __init__(self, json_dict: dict, row_index: int, groupname: str, dataset_id: str, task_id: str, test_index: int, prompt: str, expected_output: np.array):
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
        if not isinstance(expected_output, np.ndarray):
            raise ValueError(f"Expected expected_output to be a np.ndarray, but got: {expected_output}")

        self.json_dict = json_dict
        self.row_index = row_index
        self.groupname = groupname
        self.dataset_id = dataset_id
        self.task_id = task_id
        self.test_index = test_index
        self.prompt = prompt
        self.expected_output = expected_output

    @staticmethod
    def from_json_string(json_string: str, row_index: int) -> 'TaskToPromptItem':
        if not isinstance(json_string, str):
            raise ValueError(f"Expected json_string to be a string, but got: {json_string}")
        if not isinstance(row_index, int):
            raise ValueError(f"Expected row_index to be an int, but got: {row_index}")
        
        dict = json.loads(json_string)
        expected_output_json = dict['expected_output']
        expected_output_list_of_list_of_int = json.loads(expected_output_json)
        expected_output = np.array(expected_output_list_of_list_of_int, dtype=np.uint8)
        return TaskToPromptItem(
            json_dict=dict,
            row_index=row_index,
            groupname=dict['groupname'],
            dataset_id=dict['dataset'],
            task_id=dict['task'],
            test_index=dict['test_index'],
            prompt=dict['prompt'],
            expected_output=expected_output
        )

    @staticmethod
    def load_json_file(jsonl_path: str, show: bool, truncate: Optional[int] = None) -> list['TaskToPromptItem']:
        item_list = []
        with open(jsonl_file, 'r') as f:
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
        return f"TaskToPromptItem(row={self.row_index} task_id={self.task_id}, prompt.len={len(self.prompt)})"

    def __repr__(self):
        return self.__str__()

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241221_150505/task_to_prompt.jsonl'
task_to_prompt_item_list = TaskToPromptItem.load_json_file(jsonl_file, show=True, truncate=5)

print(f"Number of task_to_prompt_item_list: {len(task_to_prompt_item_list)}")

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

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

    print("DONE")
    break
