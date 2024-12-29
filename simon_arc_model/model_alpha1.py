"""
Version numbers use greek letters. Where `Alpha` is the 1st letter in the greek alphabet.
This is the 1st approach in this project. Here I used an LLM with a CodeT5 model.
The class name is `ModelAlpha1`, where `Alpha` means the 1st approach. And `1` means the 1st version.

When I'm using MPS on a M1 Mac, then inference is painfully slow. Around 40 minutes.
When I'm using CPU on a M1 Mac, then inference is fast. Around 20 minutes.
The difference is a factor of 2. I'm not sure why the difference is so large.
"""
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import torch
from safetensors.torch import load_file
from enum import Enum

class ModelAlpha1ProcessMode(Enum):
    TEMPERATURE_ZERO_BEAM5 = 'temperature_zero_beam5'
    TEMPERATURE_LOW = 'temperature_low'
    TEMPERATURE_MEDIUM = 'temperature_medium'
    TEMPERATURE_HIGH = 'temperature_high'
    TEMPERATURE_LAB1 = 'temperature_lab1'

class ModelAlpha1:
    def __init__(self, pretrained_model_name_or_path: str, input_max_length: int):
        good_input_max_length = input_max_length in {256, 512, 1024}
        if not good_input_max_length:
            # As of 2024-august-09, the model has been trained on both input_max_length 256 and 512.
            # As of 2024-september-03, the model has been trained on both input_max_length 1024.
            raise ValueError("input_max_length must be 256 or 512 or 1024")
        
        suppress_warning_about_unused_vhead_weights = True
        if suppress_warning_about_unused_vhead_weights:
            self.model = self.t5_from_pretrained_without_v_head(pretrained_model_name_or_path)
        else:
            self.model = self.t5_from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.input_max_length = input_max_length
    
    @classmethod
    def t5_from_pretrained_without_v_head(cls, pretrained_model_name_or_path: str):
        """
        The v_head.* weights are used for the PPOTrainer, and is not used for inference.
        When I load a model without v_head I get no warnings.
        however when I load a model that was trained with PPOTrainer, then I’m getting these warnings. 
        Some weights of the model checkpoint at /path/to/model/simon-arc-lab-model531 were not used when initializing T5ForConditionalGeneration: ['v_head.summary.bias', 'v_head.summary.weight']
        - This IS expected if you are initializing T5ForConditionalGeneration from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
        - This IS NOT expected if you are initializing T5ForConditionalGeneration from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
        """
        state_dict = load_file(f'{pretrained_model_name_or_path}/model.safetensors')
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('v_head.')}
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            state_dict=filtered_state_dict
        )
        return model

    @classmethod
    def t5_from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        This is the normal way to load a model.
        However when I load a model that was trained with PPOTrainer, then I’m getting warnings about unused weights.
        The v_head weights are only used by the PPOTrainer during training.
        The v_head weights are not used during inference.
        It's not a problem, but it's a annoying with the warnings. 
        """
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    def process(self, prompt: str, mode: ModelAlpha1ProcessMode) -> str:
        responses = self.process_multiple(prompt, mode, num_return_sequences=1)
        return responses[0]
    
    def process_multiple(self, prompt: str, mode: ModelAlpha1ProcessMode, num_return_sequences: int) -> list[str]:
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
        if mode == ModelAlpha1ProcessMode.TEMPERATURE_ZERO_BEAM5:
            generate_options = {
                'num_beams': 5,
                'early_stopping': True
            }
        elif mode == ModelAlpha1ProcessMode.TEMPERATURE_LOW:
            generate_options = {
                'temperature': 0.1,
                'num_beams': 5,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelAlpha1ProcessMode.TEMPERATURE_MEDIUM:
            generate_options = {
                'temperature': 0.7,
                'num_beams': 3,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelAlpha1ProcessMode.TEMPERATURE_HIGH:
            generate_options = {
                'temperature': 4.4,
                'num_beams': 3,
                'do_sample': True,
                'early_stopping': True
            }
        elif mode == ModelAlpha1ProcessMode.TEMPERATURE_LAB1:
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

        generate_options['max_length'] = 512
        generate_options['num_return_sequences'] = num_return_sequences

        outputs = self.model.generate(input_ids, **generate_options)

        # Decode all the generated sequences
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        return responses
