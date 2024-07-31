import numpy as np
import json
from typing import Optional

class Task:
    def __init__(self):
        self.input_images = []
        self.output_images = []
        self.count_examples = 0
        self.count_tests = 0

    def append_pair(self, input_image: np.array, output_image: Optional[np.array], is_example: bool):
        self.assert_count()
        if is_example and self.count_tests > 0:
            raise ValueError("Example must be added before test")
        self.input_images.append(input_image)
        self.output_images.append(output_image)
        if is_example:
            self.count_examples += 1
        else:
            self.count_tests += 1
        self.assert_count()

    def count(self) -> int:
        self.assert_count()
        return len(self.input_images)

    def assert_count(self):
        assert len(self.input_images) == len(self.output_images)
        assert self.count_examples + self.count_tests == len(self.input_images)

    def max_image_size(self) -> tuple[int, int]:
        """
        Find (width, height) of the biggest images in the task.

        Where the Output is None, the image size is not included.
        """
        self.assert_count()
        width = 0
        height = 0
        for i in range(len(self.input_images)):
            width = max(width, self.input_images[i].shape[1])
            height = max(height, self.input_images[i].shape[0])
            if self.output_images[i] is not None:
                width = max(width, self.output_images[i].shape[1])
                height = max(height, self.output_images[i].shape[0])
        return (width, height)
    
    def total_pixel_count(self) -> int:
        """
        Count the total number of pixels in all images.

        Where the Output is None, the pixel count is not included.
        """
        self.assert_count()
        count = 0
        for i in range(len(self.input_images)):
            count += self.input_images[i].shape[0] * self.input_images[i].shape[1]
            if self.output_images[i] is not None:
                count += self.output_images[i].shape[0] * self.output_images[i].shape[1]
        return count

    def to_arcagi1_dict(self) -> dict:
        array_train = []
        array_test = []
        for i in range(self.count()):
            dict = {
                'input': self.input_images[i].tolist(),
                'output': self.output_images[i].tolist(),
            }
            if i < self.count_examples:
                array_train.append(dict)
            else:
                array_test.append(dict)

        dict = {
            'train': array_train,
            'test': array_test,
        }
        return dict

    def to_arcagi1_json(self, compact: bool = False) -> str:
        """
        Convert the task to a JSON string in the ARC-AGI version1 file format.
        
        compact: If True, the JSON string will be compact without spaces.
        """
        dict = self.to_arcagi1_dict()
        if compact:
            return json.dumps(dict, separators=(',', ':'))
        else:
            return json.dumps(dict)

    def save_arcagi1_json(self, path: str, compact: bool = False):
        """
        Save the task to a JSON file in the ARC-AGI version1 file format.
        
        compact: If True, the JSON string will be compact without spaces.
        """
        with open(path, 'w') as file:
            file.write(self.to_arcagi1_json(compact))

    def show(self):
        from .task_show import task_show
        task_show(self, answer=True)
