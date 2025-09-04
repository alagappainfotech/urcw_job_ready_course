# Single line comments start with a number symbol.

""" Multiline strings can be written
    using three "s, and are often used
    as documentation.
"""

####################################################
## 1. Primitive Datatypes and Operators
## Module 1: Python Absolute Basics & Foundational Data Types
####################################################

# Execute this in python shell
# python
# >>> <Execute the code line by line here and capture the outputs>

# =============================================================================
# LEARNING OBJECTIVES:
# 1. Understand Python's design philosophy (Zen of Python)
# 2. Master basic syntax, variable assignment, and core data types
# 3. Learn fundamental operators and precedence
# 4. Develop computational thinking skills
# 5. Apply Pythonic style and PEP 8 conventions
# =============================================================================

# =============================================================================
# QUICK CHECK: Python's Design Philosophy
# The Zen of Python (PEP 20) emphasizes:
# - Readability counts
# - Simple is better than complex
# - Explicit is better than implicit
# - There should be one obvious way to do it
# =============================================================================

print("Welcome to Python! Let's explore the fundamentals.")
print("=" * 50)

# You have numbers
3  # => 3

# Math is what you would expect
1 + 1   # => 2
8 - 1   # => 7
10 * 2  # => 20
35 / 5  # => 7.0

# Floor division rounds towards negative infinity
5 // 3       # => 1
-5 // 3      # => -2
5.0 // 3.0   # => 1.0  # works on floats too
-5.0 // 3.0  # => -2.0

# The result of division is always a float
10.0 / 3  # => 3.3333333333333335

# Modulo operation
7 % 3   # => 1
# i % j have the same sign as j, unlike C
-7 % 3  # => 2

# Exponentiation (x**y, x to the yth power)
2**3  # => 8

# Enforce precedence with parentheses
1 + 3 * 2    # => 7
(1 + 3) * 2  # => 8

# Boolean values are primitives (Note: the capitalization)
True   # => True
False  # => False

# negate with not
not True   # => False
not False  # => True

# Boolean Operators
# Note "and" and "or" are case-sensitive
True and False  # => False
False or True   # => True

# True and False are actually 1 and 0 but with different keywords
True + True  # => 2
True * 8     # => 8
False - 5    # => -5

# Comparison operators look at the numerical value of True and False
0 == False   # => True
2 > True     # => True
2 == True    # => False
-5 != False  # => True

# None, 0, and empty strings/lists/dicts/tuples/sets all evaluate to False.
# All other values are True
bool(0)      # => False
bool("")     # => False
bool([])     # => False
bool({})     # => False
bool(())     # => False
bool(set())  # => False
bool(4)      # => True
bool(-6)     # => True

# Using boolean logical operators on ints casts them to booleans for evaluation,
# but their non-cast value is returned. Don't mix up with bool(ints) and bitwise
# and/or (&,|)
bool(0)   # => False
bool(2)   # => True
0 and 2   # => 0
bool(-5)  # => True
bool(2)   # => True
-5 or 0   # => -5

# Equality is ==
1 == 1  # => True
2 == 1  # => False

# Inequality is !=
1 != 1  # => False
2 != 1  # => True

# More comparisons
1 < 10  # => True
1 > 10  # => False
2 <= 2  # => True
2 >= 2  # => True

# Seeing whether a value is in a range
1 < 2 and 2 < 3  # => True
2 < 3 and 3 < 2  # => False
# Chaining makes this look nicer
1 < 2 < 3  # => True
2 < 3 < 2  # => False

# (is vs. ==) is checks if two variables refer to the same object, but == checks
# if the objects pointed to have the same values.
a = [1, 2, 3, 4]  # Point a at a new list, [1, 2, 3, 4]
b = a             # Point b at what a is pointing to
b is a            # => True, a and b refer to the same object
b == a            # => True, a's and b's objects are equal
b = [1, 2, 3, 4]  # Point b at a new list, [1, 2, 3, 4]
b is a            # => False, a and b do not refer to the same object
b == a            # => True, a's and b's objects are equal

# Strings are created with " or '
"This is a string."
'This is also a string.'

# Strings can be added too
"Hello " + "world!"  # => "Hello world!"
# String literals (but not variables) can be concatenated without using '+'
"Hello " "world!"    # => "Hello world!"

# A string can be treated like a list of characters
"Hello world!"[0]  # => 'H'

# You can find the length of a string
len("This is a string")  # => 16

# Since Python 3.6, you can use f-strings or formatted string literals.
name = "Reiko"
f"She said her name is {name}."  # => "She said her name is Reiko"
# Any valid Python expression inside these braces is returned to the string.
f"{name} is {len(name)} characters long."  # => "Reiko is 5 characters long."

# None is an object
None  # => None

# Don't use the equality "==" symbol to compare objects to None
# Use "is" instead. This checks for equality of object identity.
"etc" == None  # => False
None == None   # => True

# =============================================================================
# TRY THIS EXERCISES - Hands-on Application
# =============================================================================

print("\n" + "="*50)
print("TRY THIS EXERCISES")
print("="*50)

# Exercise 1: Getting Input and Type Conversion
print("\n1. Getting Input and Type Conversion:")
print("-" * 40)

# Try this: Get user input and convert to different types
# Uncomment the lines below to test:
# user_input = input("Enter a number: ")
# print(f"You entered: {user_input}")
# print(f"Type: {type(user_input)}")
# 
# # Convert to integer
# try:
#     number = int(user_input)
#     print(f"As integer: {number}")
#     print(f"Type: {type(number)}")
# except ValueError:
#     print("Cannot convert to integer!")

# Exercise 2: Variable Manipulation
print("\n2. Variable Manipulation:")
print("-" * 40)

# Practice binding values to variables
pi = 3.14159
radius = 2.2
area = pi * radius ** 2
print(f"Circle with radius {radius} has area: {area}")

# Observe how changes to one variable don't automatically update others
meters = 200
feet = meters * 3.28
print(f"{meters} meters = {feet} feet")

# Now change meters - feet doesn't automatically update
meters = 300
print(f"After changing meters to {meters}, feet is still: {feet}")
print("We need to recalculate feet!")

# Exercise 3: String Operations
print("\n3. String Operations:")
print("-" * 40)

# Combine strings using + and * operators
greeting = "Hello"
name = "Python"
message = greeting + " " + name + "!"
print(message)

# String repetition
separator = "-" * 20
print(separator)

# Escape sequences
print("Line 1\nLine 2\tTabbed")
print("He said \"Hello World!\"")
print('She said \'Python is great!\'')

# Triple-quoted strings for multi-line text
multi_line = """This is a
multi-line string
that spans several lines."""
print(multi_line)

# Exercise 4: Value Swapping
print("\n4. Value Swapping:")
print("-" * 40)

# Swap values without direct assignment
x = 10
y = 20
print(f"Before swap: x = {x}, y = {y}")

# Method 1: Using a temporary variable
temp = x
x = y
y = temp
print(f"After swap (method 1): x = {x}, y = {y}")

# Method 2: Python's elegant tuple unpacking
x, y = y, x
print(f"After swap (method 2): x = {x}, y = {y}")

# =============================================================================
# QUICK CHECKS - Immediate Reinforcement
# =============================================================================

print("\n" + "="*50)
print("QUICK CHECKS")
print("="*50)

# Quick Check 1: Type Identification
print("\n1. Type Identification:")
print("-" * 40)

objects = [1234, 8.99, 9.0, True, False, "hello", None]
for obj in objects:
    print(f"{obj} -> {type(obj).__name__}")

# Quick Check 2: Boolean Logic
print("\n2. Boolean Logic:")
print("-" * 40)

test_values = [0, 1, -1, "", "hello", [], [1, 2], {}, {"key": "value"}]
for value in test_values:
    print(f"bool({value}) = {bool(value)}")

# Quick Check 3: Operator Precedence
print("\n3. Operator Precedence:")
print("-" * 40)

# Predict the results before running
expressions = [
    "2 + 3 * 4",
    "(2 + 3) * 4", 
    "2 ** 3 ** 2",
    "(2 ** 3) ** 2",
    "10 / 3 * 3",
    "10 // 3 * 3"
]

for expr in expressions:
    result = eval(expr)
    print(f"{expr} = {result}")

# =============================================================================
# LAB PROBLEMS - Critical Thinking
# =============================================================================

print("\n" + "="*50)
print("LAB PROBLEMS")
print("="*50)

# Lab Problem 1: Temperature Converter
print("\n1. Temperature Converter:")
print("-" * 40)

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

# Test the functions
c_temp = 25
f_temp = celsius_to_fahrenheit(c_temp)
print(f"{c_temp}°C = {f_temp}°F")

f_temp = 77
c_temp = fahrenheit_to_celsius(f_temp)
print(f"{f_temp}°F = {c_temp}°C")

# Lab Problem 2: Simple Calculator
print("\n2. Simple Calculator:")
print("-" * 40)

def simple_calculator():
    """A simple calculator with basic operations."""
    print("Simple Calculator")
    print("Operations: +, -, *, /, **")
    
    # Get input (in a real program, you'd use input())
    num1 = 10
    num2 = 3
    operation = "+"
    
    if operation == "+":
        result = num1 + num2
    elif operation == "-":
        result = num1 - num2
    elif operation == "*":
        result = num1 * num2
    elif operation == "/":
        result = num1 / num2
    elif operation == "**":
        result = num1 ** num2
    else:
        result = "Invalid operation"
    
    print(f"{num1} {operation} {num2} = {result}")

simple_calculator()

# Lab Problem 3: Grade Calculator
print("\n3. Grade Calculator:")
print("-" * 40)

def calculate_weighted_average():
    """Calculate weighted average of grades."""
    # Example grades and weights
    grades = [85, 92, 78, 96]
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
    
    weighted_sum = sum(grade * weight for grade, weight in zip(grades, weights))
    print(f"Grades: {grades}")
    print(f"Weights: {weights}")
    print(f"Weighted Average: {weighted_sum:.2f}")

calculate_weighted_average()

# Lab Problem 4: Password Validator
print("\n4. Password Validator:")
print("-" * 40)

def validate_password(password):
    """Check if password meets basic criteria."""
    criteria = {
        "length": len(password) >= 8,
        "has_uppercase": any(c.isupper() for c in password),
        "has_lowercase": any(c.islower() for c in password),
        "has_digit": any(c.isdigit() for c in password),
        "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    }
    
    print(f"Password: {password}")
    for criterion, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"  {criterion}: {status}")
    
    return all(criteria.values())

# Test password validation
test_passwords = ["password", "Password123", "Password123!", "P@ssw0rd"]
for pwd in test_passwords:
    is_valid = validate_password(pwd)
    print(f"Overall: {'Valid' if is_valid else 'Invalid'}\n")

# =============================================================================
# ASSESSMENT CRITERIA
# =============================================================================

print("\n" + "="*50)
print("ASSESSMENT CRITERIA")
print("="*50)

print("""
For all exercises and labs, your code will be evaluated on:

1. CODE QUALITY:
   - PEP 8 compliance (indentation, spacing, naming)
   - Meaningful variable and function names
   - Clear and concise code structure

2. FUNCTIONALITY:
   - Correct implementation of requirements
   - Handles edge cases appropriately
   - Produces expected outputs

3. ERROR HANDLING:
   - Graceful handling of invalid inputs
   - Appropriate use of try-except blocks
   - Clear error messages

4. DOCUMENTATION:
   - Clear comments explaining complex logic
   - Docstrings for functions
   - Inline comments where helpful

5. TESTING:
   - Verification with multiple test cases
   - Edge case testing
   - Expected vs. actual output verification
""")

print("\n" + "="*50)
print("MODULE 1 COMPLETE!")
print("Next: Module 2 - Advanced Data Structures & Control Flow")
print("="*50)