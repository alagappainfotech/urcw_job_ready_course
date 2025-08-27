# Lesson: Python Functions
# Functions are reusable blocks of code that perform a specific task.
# They help make code modular, readable, and easier to maintain.

# Defining a function
def greet(name):
    """
    This function takes a name as input and prints a greeting message.
    :param name: str
    """
    print(f"Hello, {name}!")

# Calling a function
greet("Alice")  # Output: Hello, Alice!

# Function with return value
def add_numbers(a, b):
    """
    This function takes two numbers as input and returns their sum.
    :param a: int or float
    :param b: int or float
    :return: int or float
    """
    return a + b

# Using the return value
result = add_numbers(5, 3)
print(result)  # Output: 8

# Function with default arguments
def introduce(name, age=18):
    """
    This function introduces a person with a default age of 18.
    :param name: str
    :param age: int
    """
    print(f"My name is {name} and I am {age} years old.")

introduce("Bob")  # Output: My name is Bob and I am 18 years old.
introduce("Alice", 25)  # Output: My name is Alice and I am 25 years old.

# Function with variable-length arguments (*args)
def print_numbers(*args):
    """
    This function takes a variable number of arguments and prints them.
    :param args: tuple
    """
    for num in args:
        print(num)

print_numbers(1, 2, 3, 4)  # Output: 1 2 3 4 (each on a new line)

# Function with keyword arguments (**kwargs)
def print_details(**kwargs):
    """
    This function takes keyword arguments and prints them as key-value pairs.
    :param kwargs: dict
    """
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_details(name="Alice", age=25, city="New York")
# Output:
# name: Alice
# age: 25
# city: New York

# Lambda (anonymous) functions
# These are small, one-line functions defined using the `lambda` keyword.
square = lambda x: x ** 2
print(square(4))  # Output: 16

# Exercise:
# 1. Write a function that calculates the factorial of a number.
# 2. Write a function that checks if a given string is a palindrome.
# 3. Write a lambda function to find the maximum of two numbers.