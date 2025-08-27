import unittest
from excercise.exercise_01 import (
    last_digit, first_digit, reverse_number, max_of_numbers,
    list_example, tuple_example, set_example, dict_example,
    Stack, Queue, TreeNode
)

class TestExercise01(unittest.TestCase):
    def test_last_digit(self):
        self.assertEqual(last_digit(123), 3)
        self.assertEqual(last_digit(-456), 6)
        self.assertEqual(last_digit(0), 0)

    def test_first_digit(self):
        self.assertEqual(first_digit(123), 1)
        self.assertEqual(first_digit(-456), 4)
        self.assertEqual(first_digit(0), 0)
        self.assertEqual(first_digit(9), 9)

    def test_reverse_number(self):
        self.assertEqual(reverse_number(123), 321)
        self.assertEqual(reverse_number(-456), -654)
        self.assertEqual(reverse_number(0), 0)
        self.assertEqual(reverse_number(100), 1)

    def test_max_of_numbers(self):
        self.assertEqual(max_of_numbers(1, 2, 3), 3)
        self.assertEqual(max_of_numbers(-1, -2, -3), -1)
        with self.assertRaises(ValueError):
            max_of_numbers()

    def test_list_example(self):
        self.assertEqual(list_example(), [1, 2, 3])

    def test_tuple_example(self):
        self.assertEqual(tuple_example(), (1, 2, 3))

    def test_set_example(self):
        self.assertEqual(set_example(), {1, 2, 3})

    def test_dict_example(self):
        self.assertEqual(dict_example(), {'a': 1, 'b': 2})

    def test_stack(self):
        s = Stack()
        s.push(1)
        s.push(2)
        self.assertFalse(s.is_empty())
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)
        self.assertTrue(s.is_empty())
        with self.assertRaises(IndexError):
            s.pop()

    def test_queue(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        self.assertFalse(q.is_empty())
        self.assertEqual(q.dequeue(), 1)
        self.assertEqual(q.dequeue(), 2)
        self.assertTrue(q.is_empty())
        with self.assertRaises(IndexError):
            q.dequeue()

    def test_tree(self):
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        self.assertEqual(root.value, 1)
        self.assertEqual(root.left.value, 2)
        self.assertEqual(root.right.value, 3)

if __name__ == "__main__":
    unittest.main()