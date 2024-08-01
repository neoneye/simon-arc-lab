import unittest
import numpy as np
from .task import *
from .task_formatter_rle_compact import *

class TestTaskFormatterRLECompact(unittest.TestCase):
    def test_input_ids(self):
        task = self.task_with_3examples_2tests()
        actual = TaskFormatterRLECompact(task).input_ids()
        expected = ["I0", "I1", "I2", "I3T", "I4T"]
        self.assertEqual(actual, expected)

    def test_output_ids(self):
        task = self.task_with_3examples_2tests()
        actual = TaskFormatterRLECompact(task).output_ids()
        expected = ["O0", "O1", "O2", "O3T", "O4T"]
        self.assertEqual(actual, expected)

    def test_pair_ids(self):
        task = self.task_with_3examples_2tests()
        actual = TaskFormatterRLECompact(task).pair_ids()
        expected = ["P0", "P1", "P2", "P3T", "P4T"]
        self.assertEqual(actual, expected)

    def test_to_string_1example_1test(self):
        task = Task()
        input0 = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        output0 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        input1 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
        output1 = None
        task.append_pair(input0, output0, True)
        task.append_pair(input1, output1, False)
        actual = TaskFormatterRLECompact(task).to_string()
        expected = "I0\n2 2 0,\nO0\n2 2 1,\nI1T\n2 2 2,\nO1T\nNone"
        self.assertEqual(actual, expected)

    def test_to_string_3examples_2tests(self):
        task = self.task_with_3examples_2tests()
        actual = TaskFormatterRLECompact(task).to_string()
        expected = "I0\n1 1 0\nO0\n1 1 1\nI1\n1 1 2\nO1\n1 1 3\nI2\n1 1 4\nO2\n1 1 5\nI3T\n1 1 6\nO3T\nNone\nI4T\n1 1 7\nO4T\nNone"
        self.assertEqual(actual, expected)

    def task_with_3examples_2tests(self) -> Task:
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
        return task

if __name__ == '__main__':
    unittest.main()
