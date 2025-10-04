fruits = ['apple', 'banana', 'cherry']
print(fruits)
O/p
['apple', 'banana', 'cherry']

fruits.append('orange')
print(fruits)  

O/p
['apple', 'banana', 'cherry','orange']

fruits = ['apple', 'banana', 'cherry']
fruits.remove('banana')
print(fruits)

O/p
['apple', 'cherry']


fruits = ['apple', 'banana', 'cherry']
print(fruits[0])
O/p
'apple'

coordinates = (10, 20)
print(coordinates)  
O/p
(10,20)

print(coordinates[1])  
O/p
(10)

unique_numbers = {1, 2, 3, 3}
print(unique_numbers)  
O/p
{1,2,3}
unique_numbers.add(4)
print(unique_numbers) 
O/p
{1,2,3,4}


unique_numbers.remove(2)
print(unique_numbers)
O/p
{1,3}
person = {'name': 'Alice', 'age': 25}
print(person) 
print(person['name'])

O/p
{'name': 'Alice', 'age': 25}
Alice

person['city'] = 'New York'
print(person)
O/p
{'name': 'Alice', 'age': 25, 'city': 'New York'}

text = "hello"
print(text[1])  

# Slicing
print(text[1:4])  
O/p
e
ell

nested_list = [[1, 2], [3, 4]]
print(nested_list[1][0])  
nested_dict = {'person': {'name': 'Bob', 'age': 30}}
print(nested_dict['person']['name'])

O/p
3
Bob

Exercise:::

students = [
    {"name": "Alice", "age": 20, "grade": "A"},
    {"name": "Bob", "age": 21, "grade": "B"},
    {"name": "Charlie", "age": 19, "grade": "A"}
]

# Print student information
for student in students:
    print(f"Name: {student['name']}, Age: {student['age']}, Grade: {student['grade']}")

O/p

Name: Alice, Age: 20, Grade: A
Name: Bob, Age: 21, Grade: B
Name: Charlie, Age: 19, Grade: A
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Calculate union
union_set = set1.union(set2)
print(f"Union: {union_set}")

# Calculate intersection
intersection_set = set1.intersection(set2)
print(f"Intersection: {intersection_set}")

O/p

Union: {1, 2, 3, 4, 5, 6, 7, 8}
Intersection: {4, 5}
company = {
    "HR": {
        "employees": [
            {"name": "John", "position": "Manager"},
            {"name": "Emma", "posittion": "Recruiter"}
        ]
    },
    "IT": {
        "employees": [
            {"name": "David", "position": "Developer"},
            {"name": "Sophia", "position": "Support Specialist"}
        ]
    },
 "Marketing": {
        "employees": [
            {"name": "Michael", "position": "Director"},
            {"name": "Olivia", "position": "Marketing Specialist"}
        ]
    }
}

# Print department and employee information
for department, details in company.items():
print(f"Department: {department}")
    for employee in details["employees"]:
        print(f"Name: {employee['name']}, Position: {employee['position']}")
    print()  # Empty line for better formatting

O/p
Department: HR
Name: John, Position: Manager
Name: Emma, Position: Recruiter

Department: IT
Name: David, Position: Developer
Name: Sophia, Position: Support Specialist

Department: Marketing
Name: Michael, Position: Director
Name: Olivia, Position: Marketing Specialist
try:
    result = 10 / 0  # This will raise a ZeroDivisionError
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
O/p
Error: Division by zero is not allowed.


try:
num = int("abc")  # This will raise a ValueError
except ValueError:
    print("Error: Invalid input. Please enter a number.")
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
O/p
Error: Invalid input. Please enter a number.

try:
    result = 10 / 2
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
else:
print(f"Result: {result}")
O/p
Result 5.0

try:
    file = open("example.txt", "r")
except FileNotFoundError:
    print("Error: File not found.")
finally:
    print("Execution completed.")

O/p

Error: File not found.
Execution completed.
try:
    age = -1
    if age < 0:
        raise ValueError("Age cannot be negative.")
except ValueError as e:
    print(f"Error: {e}")

O/p
Error: Age cannot be negative.


class CustomError(Exception):
    """Custom exception class."""
    pass
try:
    raise CustomError("This is a custom error.")
except CustomError as e:
    print(f"Caught custom exception: {e}")

O/p
Caught custom exception: This is a custom error.

try:
    file = open("example.txt", "r")
    print(file.read())  # Reads the entire file
    file.close()
except FileNotFoundError:
    print("Error: File not found.")

O/p
Error: File not found.
import os
if os.path.exists("example.txt"):
    print("File exists.")
else:
    print("File does not exist.")
O/p
File exits

Exercise::

with open("numbers.txt", "w") as file:
    for i in range(1, 11):
        file.write(str(i) + "\n")

print("Numbers 1 to 10 written to numbers.txt")
O/p
Numbers 1 to 10 written to numbers.txt

def count_lines(filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

filename = "numbers.txt"
line_count = count_lines(filename)
if line_count is not None:
    print(f"Number of lines in {filename}: {line_count}")

O/p
Number of lines in numbers.txt: 10

def copy_file(source_filename, destination_filename):
    try:
        with open(source_filename, "r") as source_file:
            with open(destination_filename, "w") as destination_file:
                destination_file.write(source_file.read())
        print(f"Contents of {source_filename} copied to {destination_filename}")
    except FileNotFoundError:
        print(f"File {source_filename} not found.")

source_filename = "numbers.txt"
destination_filename = "copied_numbers.txt"
copy_file(source_filename, destination_filename)
O/p

Contents of numbers.txt copied to copied_numbers.txt

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator  
def say_hello():
    print("Hello!")

say_hello()

O/p
Something is happening before the function is called.
Hello!
Something is happening after the function is called.

def my_decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print("Before the function call.")
        result = func(*args, **kwargs)
        print("After the function call.")
        return result
    return wrapper

@my_decorator_with_args
def add(a, b):
    return a + b

print(add(3, 5))
O/p
Before the function call.
After the function call.
8

def decorator_one(func):
    def wrapper():
print("Decorator One")
        func()
    return wrapper

def decorator_two(func):
    def wrapper():
        print("Decorator Two")
        func()
    return wrapper
@decorator_one
@decorator_two
def greet():
    print("Hello!")

greet()

O/p
Decorator One
Decorator Two
Hello!
def log_method(func):
    def wrapper(*args, **kwargs):
        print(f"Method {func._name_} is called.")
        return func(*args, **kwargs)
    return wrapper

class MyClass:
    @log_method
    def display(self):
        print("This is a method in MyClass.")

obj = MyClass()
obj.display()

O/p
Method display is called.
This is a method in MyClass.

class Example:
 @staticmethod
    def static_method():
        print("This is a static method.")

    @classmethod
    def class_method(cls):
        print("This is a class method.")

    @property
def read_only_property(self):
        return "This is a read-only property."

example = Example()
example.static_method()  # Output: This is a static method.
example.class_method()   # Output: This is a class method.
print(example.read_only_property)  

O/p
This is a static method.
This is a class method.
This is a read-only property.

import time
import functools

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func._name_} executed in {execution_time:.4f} seconds.")
        return result
    return wrapper
@log_execution_time
def example_function():
    time.sleep(1)  # Simulate some time-consuming operation
    print("Function executed.")

example_function()

O/p
Function executed.
Function example_function executed in 1.0005 seconds.

def requires_authentication(func):
    def wrapper(*args, **kwargs):
        # Simulate authentication check
        user_authenticated = kwargs.get("authenticated", False)
        if not user_authenticated:
            raise Exception("User is not authenticated.")
        return func(*args, **kwargs)
    return wrapper

@requires_authentication
def protected_function(authenticated=False):
    print("User is authenticated. Function executed.")

try:
    protected_function(authenticated=True)
    protected_function(authenticated=False)
except Exception as e:
    print(e)
O/p
User is authenticated. Function executed.
User is not authenticated.


def double_numbers(iterable):
    for i in iterable:
        yield i + i

# Generators are memory-efficient because they only load the data needed to
# process the next value in the iterable. This allows them to perform
# operations on otherwise prohibitively large value ranges.
# NOTE: range replaces xrange in Python 3.
for i in double_numbers(range(1, 900000000)):  # range is a generator.
    print(i)
    if i >= 30:
        break

# Just as you can create a list comprehension, you can create generator
# comprehensions as well.
values = (-x for x in [1,2,3,4,5])
for x in values:
    print(x)  # prints -1 -2 -3 -4 -5 to console/terminal

# You can also cast a generator comprehension directly to a list.
values = (-x for x in [1,2,3,4,5])
gen_to_list = list(values)
print(gen_to_list)  # => [-1, -2, -3, -4, -5]

# Decorators are a form of syntactic sugar.
# They make code easier to read while accomplishing clunky syntax.

# Wrappers are one type of decorator.
# They're really useful for adding logging to existing functions without needing to modify them.

def log_function(func):
    def wrapper(*args, **kwargs):
        print("Entering function", func._name_)
        result = func(*args, **kwargs)
        print("Exiting function", func._name_)
        return result
    return wrapper

@log_function               # equivalent:
def my_function(x,y):       # def my_function(x,y):
    return x+y              #   return x+y
                            # my_function = log_function(my_function)
# The decorator @log_function tells us as we begin reading the function definition
# for my_function that this function will be wrapped with log_function.
# When function definitions are long, it can be hard to parse the non-decorated
# assignment at the end of the definition.

my_function(1,2)  # => "Entering function my_function"
                  # => "3"
                  # => "Exiting function my_function"

# But there's a problem.
# What happens if we try to get some information about my_function?

print(my_function._name_)  # => 'wrapper'
print(my_function._code_.co_argcount)  # => 0. The argcount is 0 because both arguments in wrapper()'s signature are optional.
# Because our decorator is equivalent to my_function = log_function(my_function)
# we've replaced information about my_function with information from wrapper

# Fix this using functools

from functools import wraps

def log_function(func):
    @wraps(func)  # this ensures docstring, function name, arguments list, etc. are all copied
                  # to the wrapped function - instead of being replaced with wrapper's info
    def wrapper(*args, **kwargs):
        print("Entering function", func._name_)
        result = func(*args, **kwargs)
        print("Exiting function", func._name_)
        return result
    return wrapper

@log_function
def my_function(x,y):
    return x+y

my_function(1,2)  # => "Entering function my_function"
                  # => "3"
                  # => "Exiting function my_function"

print(my_function._name_)  # => 'my_function'
print(my_function._code_.co_argcount)  

O/p
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
-1
-2
-3
-4
-5
[-1, -2, -3, -4, -5]
Entering function my_function
Exiting function my_function
wrapper
0
Entering function my_function
Exiting function my_function
my_function