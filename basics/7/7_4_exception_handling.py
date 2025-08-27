# Lesson: Python Exception Handling
# Exception handling allows you to manage errors gracefully without crashing your program.

# 1. Try-Except Block
# Use `try` to write code that might raise an exception and `except` to handle it.

try:
    result = 10 / 0  # This will raise a ZeroDivisionError
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
# Output: Error: Division by zero is not allowed.

# 2. Catching Multiple Exceptions
# You can handle different exceptions with multiple `except` blocks.

try:
    num = int("abc")  # This will raise a ValueError
except ValueError:
    print("Error: Invalid input. Please enter a number.")
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
# Output: Error: Invalid input. Please enter a number.

# 3. Using Else
# The `else` block runs if no exception occurs.

try:
    result = 10 / 2
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
else:
    print(f"Result: {result}")
# Output: Result: 5.0

# 4. Finally Block
# The `finally` block always executes, regardless of whether an exception occurred.

try:
    file = open("example.txt", "r")
except FileNotFoundError:
    print("Error: File not found.")
finally:
    print("Execution completed.")
# Output:
# Error: File not found.
# Execution completed.

# 5. Raising Exceptions
# You can raise exceptions manually using the `raise` keyword.

try:
    age = -1
    if age < 0:
        raise ValueError("Age cannot be negative.")
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Age cannot be negative.

# 6. Custom Exceptions
# You can define your own exception classes by inheriting from `Exception`.

class CustomError(Exception):
    """Custom exception class."""
    pass

try:
    raise CustomError("This is a custom error.")
except CustomError as e:
    print(f"Caught custom exception: {e}")
# Output: Caught custom exception: This is a custom error.

# Exercise:
# 1. Write a program to handle a `KeyError` when accessing a dictionary.
# 2. Create a custom exception for invalid email addresses.
# 3. Write a program that uses `finally` to close a file after reading.
# 4. Create a function that raises a `TypeError` if the input is not a string.
# 5. Implement a custom exception for validating email addresses.