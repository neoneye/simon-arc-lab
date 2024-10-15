import numpy as np
from .predict_output_base import PredictOutputBase

class PredictOutputDoNothing(PredictOutputBase):
    def prompt(self) -> str:
        return 'do nothing'

    def execute(self, context: dict):
        print('do nothing')

    def predicted_image(self) -> np.array:
        return np.zeros((1, 1), dtype=np.uint8)
    
    def name(self) -> str:
        return 'PredictOutputDoNothing'
