import math  
print(math.sqrt(16))  
O/P #4.0

from math import pi, pow
print(pi)  
print(pow(2, 3))  
O/p #3.141592653589793
#8.0

import math as m
print(m.factorial(5))  # Output: 120
O/p #120

import random
print(random.randint(1, 10))
O/p #2

Ex
1.area of reactangle
# geometry.py
def rectangle_area(length, width):
    """
    Calculate the area of a rectangle.

    Parameters:
    length (float): The length of the rectangle.
    width (float): The width of the rectangle.

    Returns:
    float: The area of the rectangle.
    """
    return length * width


import unittest

# Example: Testing a function
def add(a, b):
    return a + b

class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)  # Test if 2 + 3 equals 5
        self.assertNotEqual(add(2, 2), 5)  # Test if 2 + 2 is not 5

if _name_ == "_main_":
    unittest.main()
O/p #.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

from unittest.mock import MagicMock

# Example: Mocking a function
class Service:
    def fetch_data(self):
        return "Real Data"

service = Service()
service.fetch_data = MagicMock(return_value="Mocked Data")
print(service.fetch_data())  # Output: Mocked Data

O/p 
#mocked data