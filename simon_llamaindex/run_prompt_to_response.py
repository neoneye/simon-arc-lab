import os
import json
import time
from datetime import datetime
import sys
from typing import Optional
import numpy as np
from tqdm import tqdm
from dotenv import dotenv_values
from llama_index.core.llms import ChatMessage

PROVIDER_ID = "ollama"
# PROVIDER_ID = "together"

if PROVIDER_ID == "ollama":
    from llama_index.llms.ollama import Ollama
elif PROVIDER_ID == "together":
    from llama_index.llms.together import TogetherLLM
else:
    raise ValueError(f"Unknown PROVIDER_ID: {PROVIDER_ID}")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.extract_json_from_response import extract_json_from_response
from simon_arc_lab.extract_digits_from_response import extract_digits_from_response
from simon_arc_lab.image_from_json_array import image_from_json_array
from simon_arc_lab.show_prediction_result import show_prediction_result
from simon_arc_model.track_incorrect_prediction import TrackIncorrectPrediction

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

# solves 1 puzzles with llama3.1, I had to terminate it halfway because it was taking too long
# system_prompt = """You are a meticulous ARC puzzle solver. For each puzzle, you will:

# 1. Identify patterns by systematically comparing inputs and outputs in the provided examples.
# 2. Formulate hypotheses about how inputs transform to outputs.
# 3. Test and refine these hypotheses against all training pairs.
# 4. Apply the validated rules to the test input.

# At every step, question your assumptions and verify consistency before moving on."""

# solves 4 puzzles with llama3.1, Not having a system prompt worsens the ability to solve puzzles
# system_prompt = None

# solves 14 puzzles with llama3.1
# system_prompt = "You solve ARC puzzles by carefully examining patterns in each example. Identify the rules, verify them on all examples, then solve the test input"

# solves 15 puzzles with llama3.1
# system_prompt = "You are an expert at solving ARC (Abstraction & reasoning corpus) puzzles"

# solves 18 puzzles with llama3.1
# system_prompt = "Be brief and clear in your responses"

# solves 22 puzzles with llama3.1, but leaves out the reasoning steps.
system_prompt = "Be concise"

# solves 1 puzzle with previous bad prediction
# system_prompt = "The 'maybe output' is always wrong."

# solves 1 puzzle with previous bad prediction
# system_prompt = "You are an ARC solver. Figure out whats wrong with the 'maybe output'. Then explain your reasoning and provide your own final answer. The answer must be different than the 'maybe output'."

max_prompt_length = 2000
max_response_length = 2000

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
dotenv_dict = dotenv_values(dotenv_path=dotenv_path)

#task_to_prompt_jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241221_202038_task_to_prompt_json_format/task_to_prompt.jsonl'
task_to_prompt_jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241222_033847_task_to_prompt_o3_format/task_to_prompt.jsonl'
# task_to_prompt_jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241223_162108_task_to_prompt_o3_format_with_bad_predictions/task_to_prompt.jsonl'
# task_to_prompt_jsonl_file = '/Users/neoneye/git/simon_arc_lab/run_tasks_result/20241223_203727_task_to_prompt_o3_format_with_maxlimit/task_to_prompt.jsonl'
task_to_prompt_item_list = TaskToPromptItem.load_json_file(task_to_prompt_jsonl_file, show=True, truncate=None)

# remove items with too long prompt
if True:
    count_before = len(task_to_prompt_item_list)
    task_to_prompt_item_list = [item for item in task_to_prompt_item_list if len(item.prompt) <= max_prompt_length]
    count_diff = count_before - len(task_to_prompt_item_list)
    print(f"Ignoring {count_diff} items with too long prompt, exceeding: {max_prompt_length}")

print(f"Number of task_to_prompt_item_list: {len(task_to_prompt_item_list)}")

arc_bad_prediction_file = '/Users/neoneye/nobackup/git/arc-bad-prediction/data.jsonl'
track_incorrect_prediction = TrackIncorrectPrediction.load_from_jsonl(arc_bad_prediction_file)
# exit()

llm = None
model = None
if PROVIDER_ID == "ollama":
    model = "llama3.1:latest"
    # model = "qwen2.5-coder:latest"
    llm = Ollama(model=model, request_timeout=120.0, temperature=0.0)
elif PROVIDER_ID == "together":
    # model = "Qwen/QwQ-32B-Preview"
    # model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # model = "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct"
    model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    llm = TogetherLLM(
        model=model, 
        request_timeout=120.0, 
        temperature=0.1,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        max_tokens=1024
    )
else:
    raise ValueError(f"Unknown PROVIDER_ID: {PROVIDER_ID}")

# Take a snapshot of the LLM configuration, before assigning API keys, so they don't get leaked
llm_dict_as_json_string = json.dumps(llm.dict())
compact_model_config_string = f"model={llm.model} temperature={llm.temperature}"
#print(f"model name: {compact_model_config_string}")
# print(f"llm_dict_as_json_string: {llm_dict_as_json_string}")

if PROVIDER_ID == "together":
    llm.api_key = dotenv_dict['TOGETHER_API_KEY']

save_dir = f'run_tasks_result/{run_id}/'
os.makedirs(save_dir, exist_ok=True)

verbose = False
# verbose = True

correct_count = 0
pbar_posfix_dict = {
    'correct': 0,
    'task': 'none',
    'response': 0,
}
pbar = tqdm(task_to_prompt_item_list, desc=f"Processing", dynamic_ncols=True, leave=False)
for item in pbar:
    pbar_posfix_dict['task'] = item.task_id
    pbar.set_postfix(pbar_posfix_dict)

    prompt = item.prompt

    error_message_list = []

    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(
            role="system", content=system_prompt
        ))
    messages.append(ChatMessage(role="user", content=prompt))

    resp = llm.stream_chat(messages)

    if verbose:
        print("stream response:")

    start_time = time.time()
    text_items = []
    for response_index, r in enumerate(resp):
        text_item = r.delta
        text_items.append(text_item)

        # show response_index in the progress bar
        pbar_posfix_dict['response'] = response_index+1
        pbar.set_postfix(pbar_posfix_dict)

        if verbose:
            print(text_item, end="")

        if response_index >= max_response_length:
            error_message = "... Too long response, skipping the rest"
            if verbose:
                print(error_message)
            text_items.append(error_message)
            error_message_list.append(f"ERROR: aborting response, the response length has reached max_response_length: {max_response_length}")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    response_text = "".join(text_items)

    # count bytes
    response_byte_count = len(response_text.encode('utf-8'))
    response_item_count = len(text_items)

    if verbose:
        print(f"\n\nelapsed: {elapsed_time:.2f} seconds")
        print("\n\nfull response:")

    if verbose:
        print(response_text)

    response_json = None
    if False:
        try:
            response_json = extract_json_from_response(response_text)
        except Exception as e:
            response_json = None
            error_message_list.append(f"ERROR: extract_json_from_response returned {e}")
            if verbose:
                print(f"Error: extract_json_from_response returned {e}")

    if True:
        try:
            response_json = extract_digits_from_response(response_text)
        except Exception as e:
            response_json = None
            error_message_list.append(f"ERROR: extract_digits_from_response returned {e}")
            if verbose:
                print(f"Error: extract_digits_from_response returned {e}")

    if verbose:
        print(f"response_json: {response_json}")
    try:
        predicted_output_image = image_from_json_array(response_json, padding=255)
    except Exception as e:
        predicted_output_image = None
        error_message_list.append(f"ERROR: image_from_json_array returned {e}")
        if verbose:
            print(f"Error: image_from_json_array returned {e}")

    if predicted_output_image is not None:
        if verbose:
            print(f"image: {predicted_output_image.tolist()}")

    if len(error_message_list) > 0:
        predicted_output_image = None

    if predicted_output_image is not None:
        is_correct = np.array_equal(item.test_output, predicted_output_image)
    else:
        is_correct = False

    if is_correct:
        correct_count += 1
        pbar_posfix_dict['correct'] = correct_count
        pbar.set_postfix(pbar_posfix_dict)

    status = "correct" if is_correct else "incorrect"

    filename_items_optional = [
        item.groupname,
        item.task_id,
        f'test{item.test_index}',
        f'row{item.row_index}',
        status,
    ]
    filename_items = [item for item in filename_items_optional if item is not None]
    image_filename = '_'.join(filename_items) + '.png'
    chat_filename = '_'.join(filename_items) + '.md'

    image_save_path = os.path.join(save_dir, image_filename)
    chat_save_path = os.path.join(save_dir, chat_filename)

    title = f"{item.task_id} test_index={item.test_index} {status}"
    if predicted_output_image is not None:
        show_prediction_result(item.test_input, predicted_output_image, item.test_output, title, show_grid=True, save_path=image_save_path)

    chat_lines = []
    chat_lines.append(f"datasource file: {task_to_prompt_jsonl_file}")
    chat_lines.append(f"datasource row: {item.row_index}")
    chat_lines.append(f"groupname: {item.groupname}")
    chat_lines.append(f"dataset_id: {item.dataset_id}")
    chat_lines.append(f"task_id: {item.task_id}")
    chat_lines.append(f"test_index: {item.test_index}")
    chat_lines.append(f"max_prompt_length: {max_prompt_length}")
    chat_lines.append(f"max_response_length: {max_response_length}")
    chat_lines.append("")
    chat_lines.append("LLM:")
    chat_lines.append(llm_dict_as_json_string)
    chat_lines.append("")
    chat_lines.append("SYSTEM PROMPT:")
    chat_lines.append(str(system_prompt))
    chat_lines.append("")
    chat_lines.append("---")
    chat_lines.append("")
    chat_lines.append("PROMPT:")
    chat_lines.append(item.prompt)
    chat_lines.append("")
    chat_lines.append("---")
    chat_lines.append("")
    chat_lines.append("RESPONSE:")
    chat_lines.append(response_text)
    chat_lines.append("")
    chat_lines.append("---")
    chat_lines.append("")
    if len(error_message_list) > 0:
        chat_lines.append("ERROR:")
        for error in error_message_list:
            chat_lines.append(error)
        chat_lines.append("")
    chat_lines.append("")
    chat_lines.append(f"response byte count: {response_byte_count}")
    chat_lines.append(f"response item count: {response_item_count}")
    chat_lines.append(f"elapsed: {elapsed_time:.2f} seconds")
    chat_lines.append("")
    chat_lines.append("expected output:")
    chat_lines.append(f"{item.test_output.tolist()}")
    chat_lines.append("")
    chat_lines.append("predicted output:")
    if predicted_output_image is not None:
        chat_lines.append(f"{predicted_output_image.tolist()}")
    else:
        chat_lines.append("None")
    chat_lines.append("")
    chat_lines.append(f"status: {status}")
    chat_lines.append("")
    chat_content = "\n".join(chat_lines)
    with open(chat_save_path, 'w') as f:
        f.write(chat_content)

    metadata = f"run={run_id} {compact_model_config_string} elapsed={elapsed_time:.2f} response_item_count={response_item_count} response_byte_count={response_byte_count}"
    track_incorrect_prediction.track_incorrect_prediction_with_raw_data(
        item.dataset_id, 
        item.task_id, 
        item.test_index, 
        item.test_input, 
        item.test_output, 
        predicted_output_image, 
        metadata
    )
