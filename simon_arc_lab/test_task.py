import unittest
import json
import numpy as np
from .task import *

class TestTask(unittest.TestCase):
    def test_append_pair_raise_exception(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, image, False)
        with self.assertRaises(ValueError):
            task.append_pair(image, image, True)

    def test_append_pair_clones_image(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, image, False)
        # The task is unaffected by changing the original image
        image[0][0] = 255
        actual = task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0]],"output":[[0]]}],"test":[{"input":[[0]],"output":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_task_clone(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, image, False)
        new_task = task.clone()
        # The new task is unaffected by changing the original image
        image[0][0] = 255
        actual = new_task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0]],"output":[[0]]}],"test":[{"input":[[0]],"output":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_task_clone_without_test_output(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, None, False)
        new_task = task.clone()
        # The new task is unaffected by changing the original image
        image[0][0] = 255
        actual = new_task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0]],"output":[[0]]}],"test":[{"input":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_task_shuffle(self):
        task = Task()
        input0  = np.array([[0, 0]], dtype=np.uint8)
        output0 = np.array([[0, 1]], dtype=np.uint8)
        input1  = np.array([[1, 0]], dtype=np.uint8)
        output1 = np.array([[1, 1]], dtype=np.uint8)
        input2  = np.array([[2, 0]], dtype=np.uint8)
        output2 = np.array([[2, 1]], dtype=np.uint8)
        input3  = np.array([[3, 0]], dtype=np.uint8)
        output3 = np.array([[3, 1]], dtype=np.uint8)
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, True)
        task.append_pair(input2, output2, True)
        task.append_pair(input3, output3, False)
        task.shuffle_examples(4)
        actual = task.to_arcagi1_json(compact=False)
        expected = '{"train": [{"input": [[2, 0]], "output": [[2, 1]]}, {"input": [[1, 0]], "output": [[1, 1]]}, {"input": [[0, 0]], "output": [[0, 1]]}], "test": [{"input": [[3, 0]], "output": [[3, 1]]}]}'
        self.assertEqual(actual, expected)

    def test_set_all_test_outputs_to_none(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, image, False)
        task.append_pair(image, image, False)
        task.set_all_test_outputs_to_none()
        actual = task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0]],"output":[[0]]}],"test":[{"input":[[0]]},{"input":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_to_arcagi1_json_compactfalse(self):
        task = Task()
        input0 = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        output0 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        input1 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
        output1 = np.array([[3, 3], [3, 3]], dtype=np.uint8)
        input2 = np.array([[4, 4], [4, 4]], dtype=np.uint8)
        output2 = np.array([[5, 5], [5, 5]], dtype=np.uint8)
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, True)
        task.append_pair(input2, output2, False)
        actual = task.to_arcagi1_json(compact=False)
        expected = '{"train": [{"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]}, {"input": [[2, 2], [2, 2]], "output": [[3, 3], [3, 3]]}], "test": [{"input": [[4, 4], [4, 4]], "output": [[5, 5], [5, 5]]}]}'
        self.assertEqual(actual, expected)

    def test_to_arcagi1_json_compacttrue(self):
        task = Task()
        input0 = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        output0 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        input1 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
        output1 = np.array([[3, 3], [3, 3]], dtype=np.uint8)
        input2 = np.array([[4, 4], [4, 4]], dtype=np.uint8)
        output2 = np.array([[5, 5], [5, 5]], dtype=np.uint8)
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, True)
        task.append_pair(input2, output2, False)
        actual = task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0,0],[0,0]],"output":[[1,1],[1,1]]},{"input":[[2,2],[2,2]],"output":[[3,3],[3,3]]}],"test":[{"input":[[4,4],[4,4]],"output":[[5,5],[5,5]]}]}'
        self.assertEqual(actual, expected)

    def test_to_arcagi1_json_where_input_images_are_none(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, image, True)
        task.append_pair(image, image, True)
        task.append_pair(image, image, False)
        task.input_images[0] = None
        task.input_images[1] = None
        task.input_images[2] = None
        actual = task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"output":[[0]]},{"output":[[0]]}],"test":[{"output":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_to_arcagi1_json_where_output_images_are_none(self):
        task = Task()
        image = np.array([[0]], dtype=np.uint8)
        task.append_pair(image, None, True)
        task.append_pair(image, None, True)
        task.append_pair(image, None, False)
        actual = task.to_arcagi1_json(compact=True)
        expected = '{"train":[{"input":[[0]]},{"input":[[0]]}],"test":[{"input":[[0]]}]}'
        self.assertEqual(actual, expected)

    def test_create_with_arcagi1_json(self):
        json_str = '{"train":[{"input":[[0]],"output":[[1]]},{"input":[[2]],"output":[[3]]}],"test":[{"input":[[4]],"output":[[5]]}]}'
        task = Task.create_with_arcagi1_json(json.loads(json_str))
        self.assertEqual(task.count(), 3)
        self.assertEqual(task.total_pixel_count(), 6)
        self.assertEqual(task.max_image_size(), (1, 1))

    def test_create_with_arcagi1_json_without_test_output(self):
        json_str = '{"train":[{"input":[[0]],"output":[[1]]},{"input":[[2]],"output":[[3]]}],"test":[{"input":[[4]]}]}'
        task = Task.create_with_arcagi1_json(json.loads(json_str))
        self.assertEqual(task.count(), 3)
        self.assertEqual(task.total_pixel_count(), 5)
        self.assertEqual(task.max_image_size(), (1, 1))
        self.assertEqual(task.test_output(0), None)

    def test_get_image(self):
        task = Task()
        input0 = np.array([[0]], dtype=np.uint8)
        output0 = np.array([[1]], dtype=np.uint8)
        input1 = np.array([[2]], dtype=np.uint8)
        output1 = np.array([[3]], dtype=np.uint8)
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, False)
        self.assertTrue(np.array_equal(task.example_input(0), input0))
        self.assertTrue(np.array_equal(task.example_output(0), output0))
        self.assertTrue(np.array_equal(task.test_input(0), input1))
        self.assertTrue(np.array_equal(task.test_output(0), output1))

if __name__ == '__main__':
    unittest.main()
