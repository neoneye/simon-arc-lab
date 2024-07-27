import json
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from tqdm import tqdm

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
        input_ids = self.tokenizer(input_string, return_tensors='pt').input_ids
        
        outputs = self.model.generate(input_ids)
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
        print("Correct counts:")
        for key in sorted(self.counters_correct):
            print(f"{key}: {self.counters_correct[key]}")
        print("\nIncorrect counts:")
        for key in sorted(self.counters_incorrect):
            print(f"{key}: {self.counters_incorrect[key]}")
        print('-' * 80)

    def run(self, dataset):
        chunk_size = 25
        for index, dataset_item in enumerate(tqdm(dataset, desc="Processing entries"), start=1):
            self.process_dataset_item(dataset_item)
            if index > 0 and index % chunk_size == 0:
                self.print_summary()
        
        # Print summary at the end
        print("\nFinal Summary")
        self.print_summary()

# Load dataset
with open('dataset_rle.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

model_directory = '/Users/neoneye/nobackup/git/simon-arc-lab-model78'

# Initialize runner
runner = Runner(model_directory)
runner.run(dataset)
