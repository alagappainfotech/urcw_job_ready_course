# Lesson: Python Modules
# Modules are files containing Python code (functions, classes, variables) that can be reused in other programs.

# 1. Importing a Module
# Use the `import` keyword to import a module.

import math  # Importing the built-in math module
print(math.sqrt(16))  # Output: 4.0

# 2. Importing Specific Functions or Variables
# Use `from module import` to import specific items.

from math import pi, pow
print(pi)  # Output: 3.141592653589793
print(pow(2, 3))  # Output: 8.0

# 3. Renaming a Module
# Use `as` to give a module an alias.

import math as m
print(m.factorial(5))  # Output: 120

# 4. Creating a Custom Module
# Create a Python file (e.g., `my_module.py`) with functions or variables.

# Contents of `my_module.py`:
# def greet(name):
#     return f"Hello, {name}!"
# pi = 3.14

# Importing and using the custom module:
# from my_module import greet, pi
# print(greet("Alice"))  # Output: Hello, Alice!
# print(pi)  # Output: 3.14

# 5. Using the `dir()` Function
# Use `dir()` to list all attributes and methods of a module.

print(dir(math))  # Output: List of all functions and constants in the math module

# 6. Built-in Modules
# Python comes with many built-in modules like `os`, `sys`, `random`, etc.

import random
print(random.randint(1, 10))  # Output: Random number between 1 and 10

# 7. Installing and Using External Modules
# Use `pip` to install external modules (e.g., `requests`).
# Example:
# pip install requests

# Using the `requests` module:
# import requests
# response = requests.get("https://api.github.com")
# print(response.status_code)  # Output: 200 (if successful)

# Exercise:
# 1. Create a custom module with a function to calculate the area of a rectangle.
# 2. Use the `random` module to generate a list of 5 random numbers.
# 3. Install and use an external module (e.g., `numpy`) to perform a mathematical operation.