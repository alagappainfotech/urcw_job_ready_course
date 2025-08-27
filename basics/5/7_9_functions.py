# Lesson: Python Testing Frameworks
# Testing frameworks help ensure your code works as expected by automating test execution.

# 1. Unit Testing with `unittest`
# `unittest` is a built-in Python module for writing and running tests.

import unittest

# Example: Testing a function
def add(a, b):
    return a + b

class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)  # Test if 2 + 3 equals 5
        self.assertNotEqual(add(2, 2), 5)  # Test if 2 + 2 is not 5

if __name__ == "__main__":
    unittest.main()
# Output: Displays test results (e.g., OK or failures)

# 2. Testing with `pytest`
# `pytest` is a popular third-party testing framework. Install it using `pip install pytest`.

# Example: Writing a test with `pytest`
def multiply(a, b):
    return a * b

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0

# Run `pytest` in the terminal to execute tests:
# $ pytest
# Output: Displays test results with detailed information

# 3. Mocking with `unittest.mock`
# Use `unittest.mock` to replace parts of your system under test.

from unittest.mock import MagicMock

# Example: Mocking a function
class Service:
    def fetch_data(self):
        return "Real Data"

service = Service()
service.fetch_data = MagicMock(return_value="Mocked Data")
print(service.fetch_data())  # Output: Mocked Data

# 4. Parameterized Testing with `pytest`
# Use `pytest.mark.parametrize` to test multiple inputs.

import pytest

@pytest.mark.parametrize("a, b, expected", [(2, 3, 5), (0, 0, 0), (-1, 1, 0)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# 5. Code Coverage with `coverage`
# Use the `coverage` tool to measure how much of your code is tested.
# Install it using `pip install coverage`.

# Example:
# $ coverage run -m pytest
# $ coverage report
# Output: Displays the percentage of code covered by tests

# 6. Behavior-Driven Testing with `behave`
# `behave` is a framework for writing tests in plain English. Install it using `pip install behave`.

# Example: Writing a feature file (`features/example.feature`)
# Feature: Addition
#   Scenario: Add two numbers
#     Given I have two numbers 2 and 3
#     When I add them
#     Then the result should be 5

# Example: Writing step definitions (`features/steps/example_steps.py`)
# from behave import given, when, then
# @given("I have two numbers {a} and {b}")
# def step_given_two_numbers(context, a, b):
#     context.a = int(a)
#     context.b = int(b)
# $ behave

# Exercise:
# 1. Write a `unittest` test case for a function that calculates the factorial of a number.
# 2. Use `pytest` to test a function that checks if a string is a palindrome.
# 3. Mock a database call in a class method using `unittest.mock`.
```# @when("I add them")
# def step_when_add(context):
#     context.result = context.a + context.b
# @then("the result should be {expected}")
# def step_then_result(context, expected):
#     assert context.result == int(expected)

# Run `behave` in the terminal to execute tests: