# Module 3: Control Flow and Iterables - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master Python's control flow structures (`if-elif-else`, `while`, `for`)
- Understand and implement `match-case` statements (Python 3.10+)
- Work with iterables, iterators, and generators
- Handle exceptions effectively with `try-except-else-finally`
- Use context managers for resource management
- Apply control flow patterns in real-world scenarios

## Core Concepts

### 1. Conditional Statements (`if-elif-else`)
```python
# Basic if statement
if condition:
    # code block
elif another_condition:
    # alternative code block
else:
    # default code block
```

### 2. Match-Case Statements (Python 3.10+)
```python
match value:
    case pattern1:
        # handle pattern1
    case pattern2 | pattern3:  # multiple patterns
        # handle pattern2 or pattern3
    case pattern if condition:  # guard clause
        # handle pattern with condition
    case _:  # default case
        # handle all other cases
```

### 3. Loops
- **`for` loops**: Iterate over sequences
- **`while` loops**: Repeat while condition is true
- **`break`**: Exit loop immediately
- **`continue`**: Skip to next iteration
- **`else` clause**: Execute when loop completes normally

### 4. Iterables and Iterators
- **Iterable**: Object that can be iterated over
- **Iterator**: Object that implements `__iter__()` and `__next__()`
- **Generator**: Function that yields values using `yield`

### 5. Exception Handling
```python
try:
    # risky code
except SpecificException as e:
    # handle specific exception
except (Exception1, Exception2) as e:
    # handle multiple exceptions
else:
    # execute if no exception
finally:
    # always execute
```

### 6. Context Managers
```python
with context_manager as resource:
    # use resource
# resource automatically cleaned up
```

## Advanced Topics

### Generator Functions
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```

### List Comprehensions with Conditions
```python
squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Exception Hierarchy
- `BaseException`
  - `Exception`
    - `ArithmeticError`
    - `LookupError`
    - `OSError`
    - `ValueError`
    - `TypeError`
    - `KeyError`
    - `IndexError`

## Practical Applications

### 1. Data Processing Pipeline
```python
def process_data(data):
    for item in data:
        try:
            result = transform(item)
            yield result
        except ProcessingError as e:
            log_error(e)
            continue
```

### 2. File Processing with Error Handling
```python
def safe_file_processor(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                process_line(line)
    except FileNotFoundError:
        print(f"File {filename} not found")
    except PermissionError:
        print(f"Permission denied for {filename}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

### 3. Configuration Parser
```python
def parse_config(config_dict):
    match config_dict.get('mode'):
        case 'development':
            return DevelopmentConfig()
        case 'production':
            return ProductionConfig()
        case 'testing':
            return TestingConfig()
        case _:
            raise ValueError("Invalid mode")
```

## Best Practices

1. **Use specific exception types** instead of bare `except`
2. **Prefer `for` loops** over `while` loops when possible
3. **Use context managers** for resource management
4. **Implement proper error handling** with meaningful messages
5. **Use generator expressions** for memory efficiency
6. **Follow PEP 8** for code formatting
7. **Write descriptive variable names**
8. **Add type hints** for better code documentation

## Common Pitfalls

1. **Modifying list while iterating** - use list comprehension or create new list
2. **Infinite loops** - always ensure loop condition can become false
3. **Bare except clauses** - catch specific exceptions
4. **Not closing files** - use context managers
5. **Ignoring exceptions** - handle or log appropriately

## Quick Checks

### Check 1: Control Flow
```python
# What will this print?
x = 5
if x > 3:
    print("A")
elif x > 2:
    print("B")
else:
    print("C")
```

### Check 2: Match-Case
```python
# What will this print?
command = "start"
match command:
    case "start" | "begin":
        print("Starting...")
    case "stop":
        print("Stopping...")
    case _:
        print("Unknown command")
```

### Check 3: Exception Handling
```python
# What will this print?
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Division by zero")
except Exception:
    print("Other error")
else:
    print("No error")
finally:
    print("Cleanup")
```

## Lab Problems

### Lab 1: Temperature Converter with Validation
Create a program that converts temperatures between Celsius and Fahrenheit with proper error handling.

### Lab 2: File Processor with Error Recovery
Build a file processor that handles various file types and errors gracefully.

### Lab 3: Configuration Manager
Implement a configuration manager using match-case statements for different environments.

### Lab 4: Data Pipeline with Generators
Create a data processing pipeline using generators for memory efficiency.

## AI Code Comparison
When working with AI-generated code, evaluate:
- **Error handling completeness** - are all edge cases covered?
- **Resource management** - are files/connections properly closed?
- **Performance considerations** - are generators used for large datasets?
- **Code readability** - is the control flow clear and logical?
- **Exception specificity** - are specific exceptions caught rather than generic ones?

## Next Steps
- Practice with real-world data processing scenarios
- Learn about Python's `itertools` module
- Explore advanced generator patterns
- Master context manager creation
- Study exception handling in large applications
