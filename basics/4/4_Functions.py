####################################################
## 4. Functions
## Module 3: Code Organization, Functions & Error Handling
####################################################

# =============================================================================
# LEARNING OBJECTIVES:
# 1. Structure Python programs with functions and modules effectively
# 2. Understand scoping rules and namespaces in Python
# 3. Master advanced function patterns and parameter handling
# 4. Implement robust error management using exceptions
# 5. Create reusable code components and libraries
# 6. Apply best practices for code organization and documentation
# =============================================================================

# =============================================================================
# FUNCTION DESIGN PRINCIPLES:
# - Single Responsibility: Each function should do one thing well
# - Pure Functions: Functions that don't modify external state
# - Clear Interfaces: Well-defined parameters and return values
# - Documentation: Clear docstrings and type hints
# =============================================================================

print("Module 3: Code Organization, Functions & Error Handling")
print("=" * 60)

# Use "def" to create new functions
def add(x, y):
    print("x is {} and y is {}".format(x, y))
    return x + y  # Return values with a return statement

# Calling functions with parameters
add(5, 6)  # => prints out "x is 5 and y is 6" and returns 11

# Another way to call functions is with keyword arguments
add(y=6, x=5)  # Keyword arguments can arrive in any order.

# You can define functions that take a variable number of
# positional arguments
def varargs(*args):
    return args

varargs(1, 2, 3)  # => (1, 2, 3)

# You can define functions that take a variable number of
# keyword arguments, as well
def keyword_args(**kwargs):
    return kwargs

# Let's call it to see what happens
keyword_args(big="foot", loch="ness")  # => {"big": "foot", "loch": "ness"}


# You can do both at once, if you like
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)
"""
all_the_args(1, 2, a=3, b=4) prints:
    (1, 2)
    {"a": 3, "b": 4}
"""

# When calling functions, you can do the opposite of args/kwargs!
# Use * to expand args (tuples) and use ** to expand kwargs (dictionaries).
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)            # equivalent: all_the_args(1, 2, 3, 4)
all_the_args(**kwargs)         # equivalent: all_the_args(a=3, b=4)
all_the_args(*args, **kwargs)  # equivalent: all_the_args(1, 2, 3, 4, a=3, b=4)

# Returning multiple values (with tuple assignments)
def swap(x, y):
    return y, x  # Return multiple values as a tuple without the parenthesis.
                 # (Note: parenthesis have been excluded but can be included)

x = 1
y = 2
x, y = swap(x, y)     # => x = 2, y = 1
# (x, y) = swap(x,y)  # Again the use of parenthesis is optional.

# global scope
x = 5

def set_x(num):
    # local scope begins here
    # local var x not the same as global var x
    x = num    # => 43
    print(x)   # => 43

def set_global_x(num):
    # global indicates that particular var lives in the global scope
    global x
    print(x)   # => 5
    x = num    # global var x is now set to 6
    print(x)   # => 6

set_x(43)
set_global_x(6)
"""
prints:
    43
    5
    6
"""


# Python has first class functions
def create_adder(x):
    def adder(y):
        return x + y
    return adder

add_10 = create_adder(10)
add_10(3)   # => 13

# Closures in nested functions:
# We can use the nonlocal keyword to work with variables in nested scope which shouldn't be declared in the inner functions.
def create_avg():
    total = 0
    count = 0
    def avg(n):
        nonlocal total, count
        total += n
        count += 1
        return total/count
    return avg
avg = create_avg()
avg(3)  # => 3.0
avg(5)  # (3+5)/2 => 4.0
avg(7)  # (8+7)/3 => 5.0

# There are also anonymous functions
(lambda x: x > 2)(3)                  # => True
(lambda x, y: x ** 2 + y ** 2)(2, 1)  # => 5

# There are built-in higher order functions
list(map(add_10, [1, 2, 3]))          # => [11, 12, 13]
list(map(max, [1, 2, 3], [4, 2, 1]))  # => [4, 2, 3]

list(filter(lambda x: x > 5, [3, 4, 5, 6, 7]))  # => [6, 7]

# We can use list comprehensions for nice maps and filters
# List comprehension stores the output as a list (which itself may be nested).
[add_10(i) for i in [1, 2, 3]]         # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]  # => [6, 7]

# You can construct set and dict comprehensions as well.
{x for x in "abcddeef" if x not in "abc"}  # => {'d', 'e', 'f'}
{x: x**2 for x in range(5)}  # => {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# =============================================================================
# ADVANCED FUNCTION FEATURES
# =============================================================================

print("\n" + "="*60)
print("ADVANCED FUNCTION FEATURES")
print("="*60)

# Type Hints (Python 3.5+)
def calculate_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
    
    Returns:
        The area of the rectangle
    """
    return length * width

# Function with default parameters
def greet(name: str, greeting: str = "Hello", punctuation: str = "!") -> str:
    """Create a personalized greeting."""
    return f"{greeting}, {name}{punctuation}"

print(f"Area: {calculate_area(5.0, 3.0)}")
print(f"Greeting: {greet('Alice')}")
print(f"Custom greeting: {greet('Bob', 'Hi', '!!!')}")

# =============================================================================
# EXCEPTION HANDLING
# =============================================================================

print("\n" + "="*60)
print("EXCEPTION HANDLING")
print("="*60)

def safe_divide(a: float, b: float) -> float:
    """
    Safely divide two numbers with error handling.
    
    Args:
        a: Dividend
        b: Divisor
    
    Returns:
        Result of division
    
    Raises:
        ValueError: If divisor is zero
        TypeError: If inputs are not numbers
    """
    try:
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both arguments must be numbers")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return None

# Test exception handling
print(f"10 / 3 = {safe_divide(10, 3)}")
print(f"10 / 0 = {safe_divide(10, 0)}")
print(f"'10' / 3 = {safe_divide('10', 3)}")

# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

print("\n" + "="*60)
print("CONTEXT MANAGERS")
print("="*60)

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {elapsed:.4f} seconds")

# Using context manager
with Timer("List comprehension"):
    squares = [x**2 for x in range(1000)]

# =============================================================================
# DECORATORS
# =============================================================================

print("\n" + "="*60)
print("DECORATORS")
print("="*60)

def timing_decorator(func):
    """Decorator to time function execution."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    
    return wrapper

@timing_decorator
def slow_function(n: int) -> int:
    """A function that takes some time to execute."""
    import time
    time.sleep(0.1)  # Simulate slow operation
    return sum(range(n))

result = slow_function(1000)
print(f"Result: {result}")

# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

print("\n" + "="*60)
print("GENERATOR FUNCTIONS")
print("="*60)

def fibonacci_generator(n: int):
    """
    Generate Fibonacci numbers up to n.
    
    Args:
        n: Maximum number of Fibonacci numbers to generate
    
    Yields:
        Fibonacci numbers
    """
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Using generator
print("First 10 Fibonacci numbers:")
for num in fibonacci_generator(10):
    print(num, end=" ")
print()

# =============================================================================
# MODULE CREATION EXAMPLE
# =============================================================================

print("\n" + "="*60)
print("MODULE CREATION EXAMPLE")
print("="*60)

# This would typically be in a separate file (e.g., math_utils.py)
class MathUtils:
    """A collection of mathematical utility functions."""
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of n."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

# Using the utility class
print(f"Is 17 prime? {MathUtils.is_prime(17)}")
print(f"Factorial of 5: {MathUtils.factorial(5)}")
print(f"GCD of 48 and 18: {MathUtils.gcd(48, 18)}")

# =============================================================================
# ERROR HANDLING BEST PRACTICES
# =============================================================================

print("\n" + "="*60)
print("ERROR HANDLING BEST PRACTICES")
print("="*60)

class CustomError(Exception):
    """Custom exception for our application."""
    pass

class ValidationError(CustomError):
    """Raised when input validation fails."""
    pass

def validate_age(age: int) -> bool:
    """Validate age input."""
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0:
        raise ValidationError("Age cannot be negative")
    if age > 150:
        raise ValidationError("Age seems unrealistic")
    return True

# Test validation
test_ages = [25, -5, 200, "thirty"]
for age in test_ages:
    try:
        validate_age(age)
        print(f"Age {age} is valid")
    except ValidationError as e:
        print(f"Validation error for age {age}: {e}")
    except TypeError as e:
        print(f"Type error for age {age}: {e}")

# =============================================================================
# FUNCTIONAL PROGRAMMING CONCEPTS
# =============================================================================

print("\n" + "="*60)
print("FUNCTIONAL PROGRAMMING CONCEPTS")
print("="*60)

# Higher-order functions
def apply_operation(numbers: list, operation: callable) -> list:
    """Apply an operation to a list of numbers."""
    return [operation(x) for x in numbers]

# Using higher-order functions
numbers = [1, 2, 3, 4, 5]
squared = apply_operation(numbers, lambda x: x**2)
cubed = apply_operation(numbers, lambda x: x**3)

print(f"Numbers: {numbers}")
print(f"Squared: {squared}")
print(f"Cubed: {cubed}")

# Function composition
def compose(f, g):
    """Compose two functions."""
    return lambda x: f(g(x))

# Example composition
add_one = lambda x: x + 1
multiply_by_two = lambda x: x * 2
add_one_then_double = compose(multiply_by_two, add_one)

print(f"Composition result: {add_one_then_double(5)}")

# =============================================================================
# BEST PRACTICES SUMMARY
# =============================================================================

print("\n" + "="*60)
print("BEST PRACTICES SUMMARY")
print("="*60)

print("""
1. FUNCTION DESIGN:
   - Use descriptive names that indicate what the function does
   - Keep functions small and focused on a single responsibility
   - Use type hints for better code documentation
   - Write comprehensive docstrings

2. PARAMETER HANDLING:
   - Use default parameters for optional arguments
   - Use *args and **kwargs for flexible function signatures
   - Validate input parameters when necessary
   - Document parameter types and expected values

3. ERROR HANDLING:
   - Use specific exception types
   - Provide meaningful error messages
   - Handle exceptions at the appropriate level
   - Use context managers for resource management

4. CODE ORGANIZATION:
   - Group related functions into modules
   - Use classes to organize related functionality
   - Follow the DRY (Don't Repeat Yourself) principle
   - Separate concerns into different modules

5. TESTING AND DEBUGGING:
   - Write functions that are easy to test
   - Use assertions for debugging
   - Handle edge cases appropriately
   - Document expected behavior and limitations
""")

print("\n" + "="*60)
print("MODULE 3 COMPLETE!")
print("Next: Module 4 - Python Programs & Filesystem Interaction")
print("="*60)
