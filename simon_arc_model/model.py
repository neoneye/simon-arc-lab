import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

class Model:
    def __init__(self, pretrained_model_name_or_path: str, input_max_length: int):
        good_input_max_length = input_max_length in {256, 512, 1024}
        if not good_input_max_length:
            raise ValueError("input_max_length must be 256, 512, or 1024")

        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.input_max_length = input_max_length

        # Determine the device: TPU, CUDA, MPS, or CPU
        # if 'KAGGLE_URL_BASE' in os.environ:  # Check if running in Kaggle
        #     try:
        #         import torch_xla.core.xla_model as xm
        #         self.device = xm.xla_device()
        #     except ImportError:
        #         raise EnvironmentError("torch_xla is required for TPU in Kaggle.")
        if torch.cuda.is_available():
            print("Model is using CUDA")
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     print("Model is using MPS. MPS is only available on Mac, and is it takes twice as long to do inference than cpu.")
        #     self.device = torch.device("mps")
        else:
            print("Model is using CPU")
            self.device = torch.device("cpu")

        # Move the model to the appropriate device
        self.model.to(self.device)

    def process(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            max_length=self.input_max_length,
            padding='max_length',
            truncation=True
        )

        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        # Tweaking these parameters, may yield better results:
        # num_beams=3,
        # do_sample=True,
        # temperature=0.7,
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,  # Include the attention mask
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
