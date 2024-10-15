import numpy as np

class PredictOutputBase:
    def prompt(self) -> str:
        raise NotImplementedError()

    def execute(self, context: dict):
        raise NotImplementedError()

    def predicted_image(self) -> np.array:
        raise NotImplementedError()
    
    def name(self) -> str:
        raise NotImplementedError()
