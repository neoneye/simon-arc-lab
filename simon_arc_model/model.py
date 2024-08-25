from transformers import T5ForConditionalGeneration, RobertaTokenizer

class Model:
    def __init__(self, pretrained_model_name_or_path: str, input_max_length: int):
        good_input_max_length = input_max_length == 256 or input_max_length == 512
        if not good_input_max_length:
            # As of 2024-august-09, the model has been trained on both input_max_length 256 and 512.
            raise ValueError("input_max_length must be 256 or 512")
        
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.input_max_length = input_max_length
    
    def process(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, 
            return_tensors='pt',
            max_length=self.input_max_length,
            padding='max_length',
            truncation=True
        ).input_ids

        # Tweaking these parameters, may yield better results:
        # num_beams=3,
        # do_sample=True,
        # temperature=0.7,
        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
