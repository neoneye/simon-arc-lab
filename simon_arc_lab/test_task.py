import unittest
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
