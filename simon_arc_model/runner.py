from transformers import T5ForConditionalGeneration, RobertaTokenizer
from tqdm import tqdm

class Runner:
    def __init__(self, model_directory: str, input_max_length: int):
        good_input_max_length = input_max_length == 256 or input_max_length == 512
        if not good_input_max_length:
            # As of 2024-august-04, the model has only been trained on input_max_length 256. Not yet 512.
            raise ValueError("input_max_length must be 256 or 512")
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_directory)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_directory)
        self.counters_correct = {}
        self.counters_incorrect = {}
        self.input_max_length = input_max_length
    
    def process_dataset_item(self, dataset_item: dict):
        instruction = dataset_item['instruction']
        input_data = dataset_item['input']
        expected_output = dataset_item['output']
        benchmark_id = dataset_item['benchmark']
        
        input_string = f"{instruction}\n{input_data}"
        # print("length of input_string", len(input_string))
        input_ids = self.tokenizer(
            input_string, 
            return_tensors='pt',
            max_length=self.input_max_length,
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
