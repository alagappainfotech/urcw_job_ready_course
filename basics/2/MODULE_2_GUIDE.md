# Module 2: Advanced Data Structures & Control Flow

## Learning Objectives
By the end of this module, you will be able to:
- Master Python's powerful built-in collections (lists, tuples, sets, dictionaries)
- Understand advanced control flow structures and decision-making
- Implement efficient data manipulation using comprehensions
- Apply string manipulation and formatting techniques
- Develop algorithmic thinking through complex data operations

## Core Concepts

### 2.1 Python Collections Overview
Python provides four main built-in collection types, each optimized for different use cases:

**Lists:** Ordered, mutable sequences - perfect for ordered data that changes
**Tuples:** Ordered, immutable sequences - ideal for fixed data and function returns
**Sets:** Unordered collections of unique elements - excellent for membership testing
**Dictionaries:** Key-value mappings - optimal for associative data and lookups

### 2.2 Collection Characteristics
Understanding when to use each collection type is crucial for writing efficient Python code:

| Collection | Ordered | Mutable | Indexed | Duplicates | Use Case |
|------------|---------|---------|---------|------------|----------|
| List | ✓ | ✓ | ✓ | ✓ | Dynamic sequences |
| Tuple | ✓ | ✗ | ✓ | ✓ | Fixed sequences |
| Set | ✗ | ✓ | ✗ | ✗ | Unique elements |
| Dict | ✓* | ✓ | ✓ | ✗ | Key-value pairs |

*As of Python 3.7, dictionaries maintain insertion order

### 2.3 Control Flow Mastery
Advanced control flow goes beyond simple if-else statements:
- **Conditional expressions:** Ternary operators and complex conditions
- **Loop optimization:** Efficient iteration patterns
- **Comprehensions:** Pythonic data transformation
- **Generator expressions:** Memory-efficient processing

## Key Topics

### 2.4 Lists: Dynamic Sequences
- **Creation and initialization:** Multiple ways to create lists
- **Indexing and slicing:** Accessing and modifying elements
- **List methods:** append, extend, insert, remove, pop, sort, reverse
- **List comprehensions:** Elegant list creation and transformation
- **Nested lists:** Working with multi-dimensional data

### 2.5 Tuples: Immutable Sequences
- **Tuple creation:** Packing and unpacking
- **Immutability benefits:** When and why to use tuples
- **Tuple methods:** count, index
- **Named tuples:** Structured data with named fields
- **Tuple unpacking:** Advanced assignment patterns

### 2.6 Sets: Unique Collections
- **Set operations:** Union, intersection, difference, symmetric difference
- **Set methods:** add, remove, discard, update, clear
- **Frozen sets:** Immutable sets for use as dictionary keys
- **Set comprehensions:** Creating sets from iterables
- **Membership testing:** Efficient element lookup

### 2.7 Dictionaries: Key-Value Mappings
- **Dictionary creation:** Multiple initialization methods
- **Key-value operations:** Access, update, delete
- **Dictionary methods:** keys, values, items, get, update, pop
- **Dictionary comprehensions:** Elegant dictionary creation
- **Nested dictionaries:** Complex data structures
- **Default dictionaries:** Handling missing keys gracefully

### 2.8 String Manipulation
- **String methods:** split, join, strip, find, replace, format
- **String formatting:** f-strings, format(), % formatting
- **Regular expressions:** Pattern matching and text processing
- **String encoding:** Working with different character encodings

### 2.9 Advanced Control Flow
- **Conditional statements:** if-elif-else with complex conditions
- **Match statements:** Pattern matching (Python 3.10+)
- **Loop constructs:** for, while with break, continue, else
- **Comprehensions:** List, set, dictionary, and generator comprehensions
- **Boolean logic:** Truthiness and logical operations

## Hands-on Learning Path

### Quick Checks (Immediate Reinforcement)
1. **Collection Selection:** Choose the right collection for different scenarios
2. **Comprehension Writing:** Transform data using comprehensions
3. **String Operations:** Manipulate text data effectively
4. **Control Flow Logic:** Predict execution paths in complex code

### Try This Exercises (Hands-on Application)
1. **List Manipulation:** Practice advanced list operations
2. **Dictionary Usage:** Create and manipulate key-value data
3. **Set Operations:** Perform set mathematics and membership testing
4. **String Processing:** Format and transform text data

### Lab Problems (Critical Thinking)
1. **Data Analysis:** Process and analyze structured data
2. **Text Processing:** Build a word frequency counter
3. **Inventory Management:** Create a product tracking system
4. **Student Records:** Design a grade management system

## Assessment Criteria
- **Data Structure Selection:** Appropriate choice of collections
- **Algorithm Efficiency:** Optimal use of built-in methods
- **Code Readability:** Clear, Pythonic implementations
- **Error Handling:** Graceful handling of edge cases
- **Performance:** Efficient memory and time usage

## Resources
- [Python Data Structures Documentation](https://docs.python.org/3/tutorial/datastructures.html)
- [Python String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [PEP 3132 - Extended Iterable Unpacking](https://peps.python.org/pep-3132/)
- [Python Tutor for Data Structures](https://pythontutor.com/)

## Next Steps
After completing Module 2, you'll be ready to explore code organization, functions, and error handling in Module 3. The data manipulation skills you develop here will be essential for building complex applications.

### Additional Topics (to align with requirements)
- Pattern matching with `match`/`case` (Python 3.10+): add examples for structural matching on tuples/dicts and guard clauses.
- AI solution comparison: For each lab (e.g., Examining a List, Word Counting), generate an AI solution and critique it for memory use (line-by-line vs read-all), repeated work inside loops, readability, and correctness on edge cases.

### Example: match/case
```python
def classify(x):
    match x:
        case {"type": "point", "x": xval, "y": yval} if xval == yval:
            return "diagonal point"
        case (a, b):
            return f"tuple with a={a}, b={b}"
        case str() as s if s.endswith("rejected"):
            return "ends with 'rejected'"
        case _:
            return "unknown"
```

### AI-comparison rubric
- Efficiency: avoids O(n^2) scans, uses streaming for large files.
- Clarity: meaningful names, comments/docstrings where necessary.
- Correctness: handles empty data, unusual encodings, mixed types.
- Pythonic style: comprehensions where appropriate, avoids over-complex one-liners.
