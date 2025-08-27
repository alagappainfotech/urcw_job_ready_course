####################################################
## 11. Advanced Python Concepts
####################################################

# This lesson covers advanced Python concepts that are essential for professional development
# Including decorators, generators, context managers, metaclasses, and more

# 1. Decorators

# Decorators are functions that modify the behavior of other functions
# They are a form of metaprogramming

def simple_decorator(func):
    """A simple decorator that prints before and after function execution"""
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

@simple_decorator
def greet(name):
    """A simple greeting function"""
    print(f"Hello, {name}!")

# Using the decorator
greet("Alice")

# Decorator with arguments
def repeat(times):
    """Decorator that repeats a function a specified number of times"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

# say_hello() will print "Hello!" three times

# Decorator with functools.wraps to preserve function metadata
from functools import wraps

def log_function(func):
    """Decorator that logs function calls with timing"""
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@log_function
def slow_function():
    """A function that takes some time to execute"""
    import time
    time.sleep(1)
    return "Done!"

# Class-based decorators
class Timer:
    """A class-based decorator for timing functions"""
    
    def __init__(self, func):
        self.func = func
        self.calls = 0
    
    def __call__(self, *args, **kwargs):
        self.calls += 1
        import time
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"{self.func.__name__} called {self.calls} times, took {end_time - start_time:.4f} seconds")
        return result

@Timer
def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 2. Generators and Iterators

# Generators are functions that use yield to return values one at a time
# They are memory-efficient for large datasets

def fibonacci_generator(n):
    """Generate Fibonacci numbers up to n"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using the generator
for num in fibonacci_generator(10):
    print(num, end=" ")

# Generator expressions (similar to list comprehensions)
squares = (x**2 for x in range(10))
print(list(squares))

# Generator with send() method
def counter():
    """A generator that can receive values via send()"""
    count = 0
    while True:
        received = yield count
        if received is not None:
            count = received
        else:
            count += 1

# Using the generator with send()
c = counter()
next(c)  # Start the generator
print(c.send(10))  # Set count to 10
print(next(c))     # Increment to 11

# 3. Context Managers

# Context managers handle setup and cleanup automatically
# They use the `with` statement

class FileManager:
    """A custom context manager for file operations"""
    
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Setup - called when entering the context"""
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup - called when exiting the context"""
        if self.file:
            self.file.close()
        # Return False to re-raise exceptions, True to suppress them
        return False

# Using the context manager
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')

# Context manager using contextlib
from contextlib import contextmanager

@contextmanager
def timer():
    """Context manager for timing code blocks"""
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"Execution took {end - start:.4f} seconds")

# Using the timer context manager
with timer():
    # Some code to time
    import time
    time.sleep(1)

# 4. Metaclasses

# Metaclasses are classes for classes
# They allow you to customize class creation

class Singleton(type):
    """Metaclass that creates singleton classes"""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    """A singleton database class"""
    
    def __init__(self):
        self.connection = "Database connection established"
    
    def query(self, sql):
        return f"Executing: {sql}"

# Both instances will be the same
db1 = Database()
db2 = Database()
print(db1 is db2)  # True

# 5. Descriptors

# Descriptors are objects that define __get__, __set__, or __delete__
# They allow you to customize attribute access

class ValidatedProperty:
    """A descriptor that validates property values"""
    
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be at least {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be at most {self.max_value}")
        instance.__dict__[self.name] = value

class Person:
    """A class using descriptors for validation"""
    
    age = ValidatedProperty(min_value=0, max_value=150)
    height = ValidatedProperty(min_value=0, max_value=300)
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

# 6. Abstract Base Classes

from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes"""
    
    @abstractmethod
    def area(self):
        """Calculate the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape"""
        pass

class Circle(Shape):
    """Concrete implementation of Shape"""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

# 7. Property Decorators

class Temperature:
    """A class demonstrating property decorators"""
    
    def __init__(self):
        self._celsius = 0
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit (read-only)"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature in Fahrenheit"""
        self.celsius = (value - 32) * 5/9

# 8. Class Methods and Static Methods

class Date:
    """A class demonstrating class methods and static methods"""
    
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        """Create a Date object from a string (class method)"""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)
    
    @staticmethod
    def is_valid_date(year, month, day):
        """Check if a date is valid (static method)"""
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        # Simplified validation
        return True
    
    def __str__(self):
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

# 9. Magic Methods (Special Methods)

class Vector:
    """A class demonstrating magic methods"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        return int((self.x ** 2 + self.y ** 2) ** 0.5)

# 10. Closures

def make_multiplier(factor):
    """Create a function that multiplies by a factor (closure)"""
    def multiplier(x):
        return x * factor
    return multiplier

# Using closures
double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# 11. Partial Functions

from functools import partial

def power(base, exponent):
    """Calculate base raised to the power of exponent"""
    return base ** exponent

# Create partial functions
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27

# 12. Lambda Functions

# Lambda functions are small anonymous functions
add = lambda x, y: x + y
square = lambda x: x ** 2

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 13. List Comprehensions and Generator Expressions

# List comprehension
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_squares = {x**2 for x in range(10)}

# Generator expression (memory efficient)
large_squares = (x**2 for x in range(1000000))

# 14. Advanced Decorators

def cache(func):
    """A decorator that caches function results"""
    cache_data = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache_data:
            cache_data[key] = func(*args, **kwargs)
        return cache_data[key]
    
    return wrapper

@cache
def fibonacci(n):
    """Calculate Fibonacci number with caching"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 15. Exercises

# Exercise 1: Create a retry decorator
def retry(max_attempts=3, delay=1):
    """Decorator that retries a function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Exercise 2: Create a memoization decorator
def memoize(func):
    """Decorator that memoizes function results"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper

# Exercise 3: Create a context manager for database connections
@contextmanager
def database_connection(connection_string):
    """Context manager for database connections"""
    import sqlite3
    conn = sqlite3.connect(connection_string)
    try:
        yield conn
    finally:
        conn.close()

# Exercise 4: Create a descriptor for type checking
class TypedProperty:
    """Descriptor that enforces type checking"""
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be of type {self.expected_type}")
        instance.__dict__[self.name] = value

# 16. Best Practices

"""
1. Use decorators for cross-cutting concerns (logging, caching, validation)
2. Prefer generators for large datasets to save memory
3. Use context managers for resource management
4. Use descriptors for attribute access control
5. Use abstract base classes for defining interfaces
6. Use magic methods sparingly and consistently
7. Use closures for function factories
8. Use partial functions for function specialization
9. Use lambda functions for simple operations
10. Use comprehensions for readable data transformations
"""

# Next Steps:
# - Learn about Python's asyncio for asynchronous programming
# - Understand Python's multiprocessing and threading
# - Master Python's testing frameworks (pytest, unittest)
# - Explore Python's packaging and distribution tools
# - Learn about Python's security best practices 