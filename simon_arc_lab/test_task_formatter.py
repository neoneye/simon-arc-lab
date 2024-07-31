import unittest
import numpy as np
from .task import *
from .task_formatter import *

class TestTaskFormatter(unittest.TestCase):
    def test_formatter_rle_1example_1test(self):
        task = Task()
        input0 = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        output0 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        input1 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
        output1 = None
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, False)
        actual = TaskFormatterRLE(task).to_string()
        expected = "Input 0 Example\n2 2 0,\nOutput 0 Example\n2 2 1,\nInput 1 Test\n2 2 2,\nOutput 1 Test\nNone"
        self.assertEqual(actual, expected)

    def test_formatter_rle_3examples_2tests(self):
        task = Task()
        input0 = np.array([[0]], dtype=np.uint8)
        output0 = np.array([[1]], dtype=np.uint8)
        input1 = np.array([[2]], dtype=np.uint8)
        output1 = np.array([[3]], dtype=np.uint8)
        input2 = np.array([[4]], dtype=np.uint8)
        output2 = np.array([[5]], dtype=np.uint8)
        input3 = np.array([[6]], dtype=np.uint8)
        output3 = None
        input4 = np.array([[7]], dtype=np.uint8)
        output4 = None
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, True)
        task.append_pair(input2, output2, True)
        task.append_pair(input3, output3, False)
        task.append_pair(input4, output4, False)
        actual = TaskFormatterRLE(task).to_string()
        expected = "Input 0 Example\n1 1 0\nOutput 0 Example\n1 1 1\nInput 1 Example\n1 1 2\nOutput 1 Example\n1 1 3\nInput 2 Example\n1 1 4\nOutput 2 Example\n1 1 5\nInput 3 Test\n1 1 6\nOutput 3 Test\nNone\nInput 4 Test\n1 1 7\nOutput 4 Test\nNone"
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
