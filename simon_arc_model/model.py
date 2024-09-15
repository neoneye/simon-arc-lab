# When I'm using MPS on a M1 Mac, then inference is painfully slow. Around 40 minutes.
# When I'm using CPU on a M1 Mac, then inference is fast. Around 20 minutes.
# The difference is a factor of 2. I'm not sure why the difference is so large.
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import torch
from enum import Enum

class ModelProcessMode(Enum):
    TEMPERATURE_ZERO_BEAM5 = 'temperature_zero_beam5'
    TEMPERATURE_LOW = 'temperature_low'
    TEMPERATURE_MEDIUM = 'temperature_medium'
    TEMPERATURE_HIGH = 'temperature_high'
    TEMPERATURE_LAB1 = 'temperature_lab1'

class Model:
    def __init__(self, pretrained_model_name_or_path: str, input_max_length: int):
        good_input_max_length = input_max_length in {256, 512, 1024}
        if not good_input_max_length:
            # As of 2024-august-09, the model has been trained on both input_max_length 256 and 512.
            # As of 2024-september-03, the model has been trained on both input_max_length 1024.
            raise ValueError("input_max_length must be 256 or 512 or 1024")
        
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.input_max_length = input_max_length
    
    def process(self, prompt: str, mode: ModelProcessMode) -> str:
        responses = self.process_multiple(prompt, mode, num_return_sequences=1)
        return responses[0]
    
    def process_multiple(self, prompt: str, mode: ModelProcessMode, num_return_sequences: int) -> list[str]:
        input_ids = self.tokenizer(
            prompt, 
            return_tensors='pt',
            max_length=self.input_max_length,
            padding='max_length',
            truncation=True
        ).input_ids

        # Set random seed for deterministic behavior
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        generate_options = {}
        if mode == ModelProcessMode.TEMPERATURE_ZERO_BEAM5:
            generate_options = {
                'num_beams': 5,
                'early_stopping': True
            }
        elif mode == ModelProcessMode.TEMPERATURE_LOW:
            generate_options = {
                'temperature': 0.1,
                'num_beams': 5,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelProcessMode.TEMPERATURE_MEDIUM:
            generate_options = {
                'temperature': 0.7,
                'num_beams': 3,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelProcessMode.TEMPERATURE_HIGH:
            generate_options = {
                'temperature': 4.4,
                'num_beams': 3,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelProcessMode.TEMPERATURE_LAB1:
            generate_options = {
                'temperature': 0.7,
                'num_beams': 3,
                'do_sample': True,
                'early_stopping': True,
                'top_k': 50,
                'top_p': 0.95,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        generate_options['max_length'] = 128
        generate_options['num_return_sequences'] = num_return_sequences

        outputs = self.model.generate(input_ids, **generate_options)

        # Decode all the generated sequences
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        return responses
