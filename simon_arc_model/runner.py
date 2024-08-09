from .model import Model
from tqdm import tqdm

class Runner:
    def __init__(self, model: Model):
        self.model = model
        self.counters_correct = {}
        self.counters_incorrect = {}
    
    def process_dataset_item(self, dataset_item: dict):
        instruction = dataset_item['instruction']
        input_data = dataset_item['input']
        expected_response = dataset_item['output']
        benchmark_id = dataset_item['benchmark']
        
        prompt = f"{instruction}\n{input_data}"
        actual_response = self.model.process(prompt)
        
        is_correct = actual_response == expected_response

        verbose = False        
        if verbose:
            print(f"Instruction: {instruction}")
            print(f"Input: {input_data}")
            print(f"Expected response: {expected_response}")
            print(f"Actual response: {actual_response}")
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
