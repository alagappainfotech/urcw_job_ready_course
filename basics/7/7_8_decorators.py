# Lesson: Python Decorators
# Decorators are a way to modify or extend the behavior of functions or methods without changing their code.

# 1. Basic Decorator
# A decorator is a function that takes another function as input and returns a new function.

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator  # Applying the decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Something is happening before the function is called.
# Hello!
# Something is happening after the function is called.

# 2. Decorators with Arguments
# To pass arguments to the decorated function, use *args and **kwargs.

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
# Output:
# Before the function call.
# After the function call.
# 8

# 3. Chaining Multiple Decorators
# You can apply multiple decorators to a single function.

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
# Output:
# Decorator One
# Decorator Two
# Hello!

# 4. Decorating Methods in Classes
# Decorators can also be used with methods in classes.

def log_method(func):
    def wrapper(*args, **kwargs):
        print(f"Method {func.__name__} is called.")
        return func(*args, **kwargs)
    return wrapper

class MyClass:
    @log_method
    def display(self):
        print("This is a method in MyClass.")

obj = MyClass()
obj.display()
# Output:
# Method display is called.
# This is a method in MyClass.

# 5. Built-in Decorators
# Python provides built-in decorators like @staticmethod, @classmethod, and @property.

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
print(example.read_only_property)  # Output: This is a read-only property.

# Exercise:
# 1. Create a decorator that logs the execution time of a function.
# 2. Write a decorator that checks if a user is authenticated before executing a function.
# 3. Create a class with a method that uses a custom decorator to log method calls.