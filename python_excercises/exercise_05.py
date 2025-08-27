# List
def list_example():
    return [1, 2, 3]

# Tuple
def tuple_example():
    return (1, 2, 3)

# Set
def set_example():
    return {1, 2, 3}

# Dictionary
def dict_example():
    return {'a': 1, 'b': 2}

# Stack (using list)
class Stack:
    def __init__(self):
        self._data = []
    def push(self, item):
        self._data.append(item)
    def pop(self):
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()
    def is_empty(self):
        return not self._data

# Queue (using collections.deque)
from collections import deque
class Queue:
    def __init__(self):
        self._data = deque()
    def enqueue(self, item):
        self._data.append(item)
    def dequeue(self):
        if not self._data:
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()
    def is_empty(self):
        return not self._data

# Tree (simple binary tree)
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None