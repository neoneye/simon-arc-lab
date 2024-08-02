from transformers import T5ForConditionalGeneration, RobertaTokenizer
from tqdm import tqdm
from simon_arc_lab.image_util import *
from simon_arc_lab.task import *
from simon_arc_lab.task_formatter_rle_compact import *

class Runner:
    def __init__(self, model_directory):
        self.model = T5ForConditionalGeneration.from_pretrained(model_directory)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_directory)
        self.counters_correct = {}
        self.counters_incorrect = {}
    
    def process_dataset_item(self, dataset_item):
        instruction = dataset_item['instruction']
        input_data = dataset_item['input']
        expected_output = dataset_item['output']
        benchmark_id = dataset_item['benchmark']
        
        input_string = f"{instruction}\n{input_data}"
        input_ids = self.tokenizer(
            input_string, 
            return_tensors='pt',
            max_length=256,
            padding='max_length',
            truncation=True
        ).input_ids
        
        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        generated_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        is_correct = generated_output == expected_output

        verbose = False        
        if verbose:
            print(f"Instruction: {instruction}")
            print(f"Input: {input_data}")
            print(f"Expected Output: {expected_output}")
            print(f"Generated Output: {generated_output}")
            print(f"Correct: {is_correct}")
            print('-' * 80)

        if is_correct:
            if benchmark_id in self.counters_correct:
                self.counters_correct[benchmark_id] += 1
            else:
                self.counters_correct[benchmark_id] = 1
        else:
            if benchmark_id in self.counters_incorrect:
                self.counters_incorrect[benchmark_id] += 1
            else:
                self.counters_incorrect[benchmark_id] = 1

    def print_summary(self):
        print(f"Correct sum: {sum(self.counters_correct.values())}")
        for key in sorted(self.counters_correct):
            print(f"{key}: {self.counters_correct[key]}")
        print(f"\nIncorrect sum: {sum(self.counters_incorrect.values())}")
        for key in sorted(self.counters_incorrect):
            print(f"{key}: {self.counters_incorrect[key]}")
        print('-' * 80)

    def run(self, dataset):
        chunk_size = 10
        for index, dataset_item in enumerate(tqdm(dataset, desc="Processing entries"), start=1):
            self.process_dataset_item(dataset_item)
            if index > 0 and index % chunk_size == 0:
                self.print_summary()
        
        # Print summary at the end
        print("\nFinal Summary")
        self.print_summary()

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model122'


filename = 'testdata/25ff71a9.json'
task = Task.load_arcagi1(filename)
print(task)

task_without_test_output = task.clone()
task_without_test_output.set_all_test_outputs_to_none()
# print(task_without_test_output.to_arcagi1_json(compact=True))

task_formatter = TaskFormatterRLECompact(task_without_test_output)
output_ids = task_formatter.output_ids()

dataset_name = 'SIMONSOLVETRANSLATE'

dataset_items = []
for test_index in range(task.count_tests):
    # print(f"Test {test_index}")

    input = task_formatter.to_string()

    expected_output = task.test_output(test_index)

    test_output_id = output_ids[task_without_test_output.count_examples + test_index]

    # TODO: let the model make a guess about the output_height
    output_height = 3
    for output_y in range(output_height):
        instruction = f"{dataset_name}, {test_output_id}, predict row {output_y}"

        pixel_list = image_get_row_as_list(expected_output, output_y)

        output = ''.join(map(str, pixel_list))
        benchmark_id = f'dataset={filename} test_index={test_index} output_y={output_y}'

        dataset_item = {
            'instruction': instruction,
            'input': input,
            'output': output,
            'benchmark': benchmark_id
        }
        # print(dataset_item)
        dataset_items.append(dataset_item)

# Initialize runner
runner = Runner(model_directory)
runner.run(dataset_items)
