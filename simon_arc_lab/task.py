import numpy as np
import json
from typing import Optional
import random

class Task:
    def __init__(self):
        self.input_images = []
        self.output_images = []
        self.count_examples = 0
        self.count_tests = 0
        self.metadata_task_id = None
        self.metadata_path = None

    @classmethod
    def load_arcagi1(cls, filepath: str) -> 'Task':
        """
        Load a task from a JSON file in the ARC-AGI version1 file format.
        """
        with open(filepath) as f:
            json_dict = json.load(f)
        return cls.create_with_arcagi1_json(json_dict)

    @classmethod
    def create_with_arcagi1_json(cls, json_dict: dict) -> 'Task':
        """
        Create a task from a JSON dictionary in the ARC-AGI version1 file format.
        """
        task = Task()
        for json_pair in json_dict['train']:
            input = np.array(json_pair['input'], np.uint8)
            output = np.array(json_pair['output'], np.uint8)
            task.append_pair(input, output, True)
        for json_pair in json_dict['test']:
            input = np.array(json_pair['input'], np.uint8)
            if 'output' in json_pair:
                output = np.array(json_pair['output'], np.uint8)
            else:
                output = None
            task.append_pair(input, output, False)
        return task

    def clone(self) -> 'Task':
        """
        Create a deep copy of the task.
        """
        task = Task()
        task.metadata_task_id = self.metadata_task_id
        task.metadata_path = self.metadata_path
        for i in range(self.count()):
            task.append_pair(self.input_images[i], self.output_images[i], i < self.count_examples)
        return task

    def append_pair(self, input_image: np.array, output_image: Optional[np.array], is_example: bool):
        self.assert_count()
        if is_example and self.count_tests > 0:
            raise ValueError("Example must be added before test")
        self.input_images.append(np.copy(input_image))
        if output_image is None:
            self.output_images.append(None)
        else:
            self.output_images.append(np.copy(output_image))
        if is_example:
            self.count_examples += 1
        else:
            self.count_tests += 1
        self.assert_count()

    def shuffle_examples(self, seed: int):
        """
        Shuffle the order of the examples.
        """
        new_indices = list(range(self.count_examples))
        random.Random(seed).shuffle(new_indices)
        new_input_images = [self.input_images[i] for i in new_indices]
        new_output_images = [self.output_images[i] for i in new_indices]
        self.input_images[:self.count_examples] = new_input_images
        self.output_images[:self.count_examples] = new_output_images

    def count(self) -> int:
        self.assert_count()
        return len(self.input_images)

    def assert_count(self):
        assert len(self.input_images) == len(self.output_images)
        assert self.count_examples + self.count_tests == len(self.input_images)

    def example_input(self, i: int) -> np.array:
        if i < 0 or i >= self.count_examples:
            raise ValueError("Invalid index")
        return self.input_images[i]

    def example_output(self, i: int) -> np.array:
        if i < 0 or i >= self.count_examples:
            raise ValueError("Invalid index")
        return self.output_images[i]

    def test_input(self, i: int) -> np.array:
        if i < 0 or i >= self.count_tests:
            raise ValueError("Invalid index")
        return self.input_images[self.count_examples + i]

    def test_output(self, i: int) -> Optional[np.array]:
        if i < 0 or i >= self.count_tests:
            raise ValueError("Invalid index")
        return self.output_images[self.count_examples + i]

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

    def set_all_test_outputs_to_none(self):
        """
        Erase all test outputs, so it's not possible to cheat by looking at the answers.
        """
        for i in range(self.count_tests):
            self.output_images[self.count_examples + i] = None

    def to_arcagi1_dict(self) -> dict:
        array_train = []
        array_test = []
        for i in range(self.count()):
            input_image = self.input_images[i]
            output_image = self.output_images[i]
            dict = {}
            if input_image is not None:
                dict['input'] = input_image.tolist()
            if output_image is not None:
                dict['output'] = output_image.tolist()
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

    def save_arcagi1(self, path: str, compact: bool = False):
        """
        Save the task to a JSON file in the ARC-AGI version1 file format.
        
        compact: If True, the JSON string will be compact without spaces.
        """
        with open(path, 'w') as file:
            file.write(self.to_arcagi1_json(compact))

    def show(self, show_grid: bool = True, show_answer: bool = True, save_path: Optional[str] = None):
        """
        Show the task in a graphical user interface, or save the image to a PNG file.

        Show the task in a graphical user interface:
        task.show()

        Save the image to a PNG file in current directory:
        task.show(save_path='task.png')

        Save the image to a PNG file in a specific directory:
        task.show(save_path='path/to/task.png')
        """
        from .task_show_matplotlib import task_show_matplotlib
        task_show_matplotlib(self, show_grid, show_answer, save_path)

    def __str__(self):
        if self.metadata_task_id is not None:
            task_id_pretty = f"Task: '{self.metadata_task_id}'"
        else:
            task_id_pretty = 'Task without ID'
        
        parts = [
            task_id_pretty,
            f"{self.count_examples} examples and {self.count_tests} tests",
            f'Max image size: {self.max_image_size()}',
            f'Total pixel count: {self.total_pixel_count()}',
        ]
        if self.metadata_path is not None:
            parts.append(f"path='{self.metadata_path}'")

        return '\n'.join(parts)

    def __repr__(self):
        return (f'<Task(id={self.metadata_task_id}, examples={self.count_examples}, tests={self.count_tests}, '
                f'max_image_size={self.max_image_size()}, total_pixel_count={self.total_pixel_count()}, '
                f"path='{self.metadata_path}')>")

if __name__ == '__main__':
    # How to run this snippet
    # PROMPT> python -m simon_arc_lab.task
    filename = 'testdata/ARC-AGI/data/training/25ff71a9.json'
    task = Task.load_arcagi1(filename)
    print(task)
    print(repr(task))
    task.show()

