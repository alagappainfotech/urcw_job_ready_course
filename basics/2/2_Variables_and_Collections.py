####################################################
## 2. Variables and Collections
## Module 2: Advanced Data Structures & Control Flow
####################################################

# =============================================================================
# LEARNING OBJECTIVES:
# 1. Master Python's powerful built-in collections (lists, tuples, sets, dictionaries)
# 2. Understand advanced control flow structures and decision-making
# 3. Implement efficient data manipulation using comprehensions
# 4. Apply string manipulation and formatting techniques
# 5. Develop algorithmic thinking through complex data operations
# =============================================================================

# =============================================================================
# COLLECTION CHARACTERISTICS OVERVIEW:
# | Collection | Ordered | Mutable | Indexed | Duplicates | Use Case |
# |------------|---------|---------|---------|------------|----------|
# | List       | ✓       | ✓       | ✓       | ✓          | Dynamic sequences |
# | Tuple      | ✓       | ✗       | ✓       | ✓          | Fixed sequences |
# | Set        | ✗       | ✓       | ✗       | ✗          | Unique elements |
# | Dict       | ✓*      | ✓       | ✓       | ✗          | Key-value pairs |
# *As of Python 3.7, dictionaries maintain insertion order
# =============================================================================

print("Module 2: Advanced Data Structures & Control Flow")
print("=" * 60)

# Python has a print function
print("I'm Python. Nice to meet you!")  # => I'm Python. Nice to meet you!

# By default the print function also prints out a newline at the end.
# Use the optional argument end to change the end string.
print("Hello, World", end="!")  # => Hello, World!

# Simple way to get input data from console
input_string_var = input("Enter some data: ")  # Returns the data as a string

# There are no declarations, only assignments.
# Convention in naming variables is snake_case style
some_var = 5
some_var  # => 5

# Accessing a previously unassigned variable is an exception.
# See Control Flow to learn more about exception handling.
some_unknown_var  # Raises a NameError

# if can be used as an expression
# Equivalent of C's '?:' ternary operator
"yay!" if 0 > 1 else "nay!"  # => "nay!"

# Lists store sequences
li = []
# You can start with a prefilled list
other_li = [4, 5, 6]

# Add stuff to the end of a list with append
li.append(1)    # li is now [1]
li.append(2)    # li is now [1, 2]
li.append(4)    # li is now [1, 2, 4]
li.append(3)    # li is now [1, 2, 4, 3]
# Remove from the end with pop
li.pop()        # => 3 and li is now [1, 2, 4]
# Let's put it back
li.append(3)    # li is now [1, 2, 4, 3] again.

# Access a list like you would any array
li[0]   # => 1
# Look at the last element
li[-1]  # => 3

# Looking out of bounds is an IndexError
li[4]  # Raises an IndexError

# You can look at ranges with slice syntax.
# The start index is included, the end index is not
# (It's a closed/open range for you mathy types.)
li[1:3]   # Return list from index 1 to 3 => [2, 4]
li[2:]    # Return list starting from index 2 => [4, 3]
li[:3]    # Return list from beginning until index 3  => [1, 2, 4]
li[::2]   # Return list selecting elements with a step size of 2 => [1, 4]
li[::-1]  # Return list in reverse order => [3, 4, 2, 1]
# Use any combination of these to make excercise slices
# li[start:end:step]

# Make a one layer deep copy using slices
li2 = li[:]  # => li2 = [1, 2, 4, 3] but (li2 is li) will result in false.

# Remove arbitrary elements from a list with "del"
del li[2]  # li is now [1, 2, 3]

# Remove first occurrence of a value
li.remove(2)  # li is now [1, 3]
li.remove(2)  # Raises a ValueError as 2 is not in the list

# Insert an element at a specific index
li.insert(1, 2)  # li is now [1, 2, 3] again

# Get the index of the first item found matching the argument
li.index(2)  # => 1
li.index(4)  # Raises a ValueError as 4 is not in the list

# You can add lists
# Note: values for li and for other_li are not modified.
li + other_li  # => [1, 2, 3, 4, 5, 6]

# Concatenate lists with "extend()"
li.extend(other_li)  # Now li is [1, 2, 3, 4, 5, 6]

# Check for existence in a list with "in"
1 in li  # => True

# Examine the length with "len()"
len(li)  # => 6


# Tuples are like lists but are immutable.
tup = (1, 2, 3)
tup[0]      # => 1
tup[0] = 3  # Raises a TypeError

# Note that a tuple of length one has to have a comma after the last element but
# tuples of other lengths, even zero, do not.
type((1))   # => <class 'int'>
type((1,))  # => <class 'tuple'>
type(())    # => <class 'tuple'>

# You can do most of the list operations on tuples too
len(tup)         # => 3
tup + (4, 5, 6)  # => (1, 2, 3, 4, 5, 6)
tup[:2]          # => (1, 2)
2 in tup         # => True

# You can unpack tuples (or lists) into variables
a, b, c = (1, 2, 3)  # a is now 1, b is now 2 and c is now 3
# You can also do extended unpacking
a, *b, c = (1, 2, 3, 4)  # a is now 1, b is now [2, 3] and c is now 4
# Tuples are created by default if you leave out the parentheses
d, e, f = 4, 5, 6  # tuple 4, 5, 6 is unpacked into variables d, e and f
# respectively such that d = 4, e = 5 and f = 6
# Now look how easy it is to swap two values
e, d = d, e  # d is now 5 and e is now 4


# Dictionaries store mappings from keys to values
empty_dict = {}
# Here is a prefilled dictionary
filled_dict = {"one": 1, "two": 2, "three": 3}

# Note keys for dictionaries have to be immutable types. This is to ensure that
# the key can be converted to a constant hash value for quick look-ups.
# Immutable types include ints, floats, strings, tuples.
invalid_dict = {[1,2,3]: "123"}  # => Yield a TypeError: unhashable type: 'list'
valid_dict = {(1,2,3):[1,2,3]}   # Values can be of any type, however.

# Look up values with []
filled_dict["one"]  # => 1

# Get all keys as an iterable with "keys()". We need to wrap the call in list()
# to turn it into a list. We'll talk about those later.  Note - for Python
# versions <3.7, dictionary key ordering is not guaranteed. Your results might
# not match the example below exactly. However, as of Python 3.7, dictionary
# items maintain the order at which they are inserted into the dictionary.
list(filled_dict.keys())  # => ["three", "two", "one"] in Python <3.7
list(filled_dict.keys())  # => ["one", "two", "three"] in Python 3.7+


# Get all values as an iterable with "values()". Once again we need to wrap it
# in list() to get it out of the iterable. Note - Same as above regarding key
# ordering.
list(filled_dict.values())  # => [3, 2, 1]  in Python <3.7
list(filled_dict.values())  # => [1, 2, 3] in Python 3.7+

# Check for existence of keys in a dictionary with "in"
"one" in filled_dict  # => True
1 in filled_dict      # => False

# Looking up a non-existing key is a KeyError
filled_dict["four"]  # KeyError

# Use "get()" method to avoid the KeyError
filled_dict.get("one")      # => 1
filled_dict.get("four")     # => None
# The get method supports a default argument when the value is missing
filled_dict.get("one", 4)   # => 1
filled_dict.get("four", 4)  # => 4

# "setdefault()" inserts into a dictionary only if the given key isn't present
filled_dict.setdefault("five", 5)  # filled_dict["five"] is set to 5
filled_dict.setdefault("five", 6)  # filled_dict["five"] is still 5

# Adding to a dictionary
filled_dict.update({"four":4})  # => {"one": 1, "two": 2, "three": 3, "four": 4}
filled_dict["four"] = 4         # another way to add to dict

# Remove keys from a dictionary with del
del filled_dict["one"]  # Removes the key "one" from filled dict

# From Python 3.5 you can also use the additional unpacking options
{"a": 1, **{"b": 2}}  # => {'a': 1, 'b': 2}
{"a": 1, **{"a": 2}}  # => {'a': 2}


# Sets store ... well sets
empty_set = set()
# Initialize a set with a bunch of values.
some_set = {1, 1, 2, 2, 3, 4}  # some_set is now {1, 2, 3, 4}

# Similar to keys of a dictionary, elements of a set have to be immutable.
invalid_set = {[1], 1}  # => Raises a TypeError: unhashable type: 'list'
valid_set = {(1,), 1}

# Add one more item to the set
filled_set = some_set
filled_set.add(5)  # filled_set is now {1, 2, 3, 4, 5}
# Sets do not have duplicate elements
filled_set.add(5)  # it remains as before {1, 2, 3, 4, 5}

# Do set intersection with &
other_set = {3, 4, 5, 6}
filled_set & other_set  # => {3, 4, 5}

# Do set union with |
filled_set | other_set  # => {1, 2, 3, 4, 5, 6}

# Do set difference with -
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}

# Do set symmetric difference with ^
{1, 2, 3, 4} ^ {2, 3, 5}  # => {1, 4, 5}

# Check if set on the left is a superset of set on the right
{1, 2} >= {1, 2, 3}  # => False

# Check if set on the left is a subset of set on the right
{1, 2} <= {1, 2, 3}  # => True

# Check for existence in a set with in
2 in filled_set   # => True
10 in filled_set  # => False

# Make a one layer deep copy
filled_set = some_set.copy()  # filled_set is {1, 2, 3, 4, 5}
filled_set is some_set        # => False

# =============================================================================
# ADVANCED COLLECTION OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("ADVANCED COLLECTION OPERATIONS")
print("="*60)

# List Comprehensions - Pythonic way to create lists
print("\n1. List Comprehensions:")
print("-" * 40)

# Basic list comprehension
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# List comprehension with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# Nested list comprehension
matrix = [[i + j for j in range(3)] for i in range(3)]
print(f"Matrix: {matrix}")

# Dictionary Comprehensions
print("\n2. Dictionary Comprehensions:")
print("-" * 40)

# Basic dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
print(f"Square dictionary: {square_dict}")

# Dictionary comprehension with condition
even_square_dict = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"Even square dictionary: {even_square_dict}")

# Set Comprehensions
print("\n3. Set Comprehensions:")
print("-" * 40)

# Basic set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "python", "code"]}
print(f"Unique word lengths: {unique_lengths}")

# =============================================================================
# STRING MANIPULATION AND FORMATTING
# =============================================================================

print("\n" + "="*60)
print("STRING MANIPULATION AND FORMATTING")
print("="*60)

# String Methods
print("\n1. String Methods:")
print("-" * 40)

text = "  Hello, World!  "
print(f"Original: '{text}'")
print(f"Strip: '{text.strip()}'")
print(f"Lower: '{text.lower()}'")
print(f"Upper: '{text.upper()}'")
print(f"Replace: '{text.replace('World', 'Python')}'")
print(f"Split: {text.split(',')}")
print(f"Startswith 'Hello': {text.strip().startswith('Hello')}")
print(f"Endswith '!': {text.strip().endswith('!')}")

# String Formatting
print("\n2. String Formatting:")
print("-" * 40)

name = "Alice"
age = 25
score = 87.5

# f-strings (Python 3.6+)
print(f"f-string: {name} is {age} years old and scored {score:.1f}%")

# format() method
print("format(): {} is {} years old and scored {:.1f}%".format(name, age, score))

# % formatting (older style)
print("% formatting: %s is %d years old and scored %.1f%%" % (name, age, score))

# String joining
print("\n3. String Joining:")
print("-" * 40)

words = ["Python", "is", "awesome", "for", "data", "science"]
sentence = " ".join(words)
print(f"Joined: {sentence}")

# Custom separator
csv_line = ",".join(["Alice", "25", "87.5", "Computer Science"])
print(f"CSV format: {csv_line}")

# =============================================================================
# CONTROL FLOW WITH COLLECTIONS
# =============================================================================

print("\n" + "="*60)
print("CONTROL FLOW WITH COLLECTIONS")
print("="*60)

# Conditional expressions with collections
print("\n1. Conditional Expressions:")
print("-" * 40)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# List comprehension with conditional
even_numbers = [x for x in numbers if x % 2 == 0]
odd_numbers = [x for x in numbers if x % 2 != 0]
print(f"Even numbers: {even_numbers}")
print(f"Odd numbers: {odd_numbers}")

# Complex conditions
filtered_numbers = [x for x in numbers if x > 3 and x < 8]
print(f"Numbers between 3 and 8: {filtered_numbers}")

# Nested loops and conditions
print("\n2. Nested Operations:")
print("-" * 40)

# Flatten a nested list
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested_list for item in sublist]
print(f"Nested list: {nested_list}")
print(f"Flattened: {flattened}")

# =============================================================================
# PRACTICAL EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("PRACTICAL EXAMPLES")
print("="*60)

# Example 1: Data Processing
print("\n1. Data Processing Example:")
print("-" * 40)

# Sample data: student grades
student_data = [
    {"name": "Alice", "grades": [85, 92, 78, 96]},
    {"name": "Bob", "grades": [90, 88, 95, 87]},
    {"name": "Charlie", "grades": [70, 75, 80, 72]},
    {"name": "Diana", "grades": [95, 98, 92, 94]}
]

# Calculate averages using list comprehension
for student in student_data:
    name = student["name"]
    grades = student["grades"]
    average = sum(grades) / len(grades)
    print(f"{name}: {grades} -> Average: {average:.1f}")

# Find students with average > 85
high_achievers = [s["name"] for s in student_data 
                 if sum(s["grades"]) / len(s["grades"]) > 85]
print(f"High achievers (>85): {high_achievers}")

# Example 2: Text Analysis
print("\n2. Text Analysis Example:")
print("-" * 40)

text = "Python is a powerful programming language. Python is used for data science, web development, and automation."

# Word frequency analysis
words = text.lower().replace(",", "").replace(".", "").split()
word_freq = {}
for word in words:
    word_freq[word] = word_freq.get(word, 0) + 1

print("Word frequency:")
for word, count in sorted(word_freq.items()):
    print(f"  {word}: {count}")

# Most common words
most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"Most common words: {most_common}")

# Example 3: Set Operations for Data Analysis
print("\n3. Set Operations Example:")
print("-" * 40)

# Two groups of students
group_a = {"Alice", "Bob", "Charlie", "Diana", "Eve"}
group_b = {"Bob", "Charlie", "Frank", "Grace", "Henry"}

print(f"Group A: {group_a}")
print(f"Group B: {group_b}")

# Set operations
both_groups = group_a & group_b  # Intersection
only_a = group_a - group_b       # Difference
only_b = group_b - group_a       # Difference
all_students = group_a | group_b # Union

print(f"Students in both groups: {both_groups}")
print(f"Students only in Group A: {only_a}")
print(f"Students only in Group B: {only_b}")
print(f"All students: {all_students}")

# =============================================================================
# PERFORMANCE CONSIDERATIONS
# =============================================================================

print("\n" + "="*60)
print("PERFORMANCE CONSIDERATIONS")
print("="*60)

print("""
Collection Performance Characteristics:

1. LISTS:
   - Access by index: O(1)
   - Search: O(n)
   - Insert/Delete at end: O(1)
   - Insert/Delete at beginning: O(n)

2. TUPLES:
   - Access by index: O(1)
   - Search: O(n)
   - Immutable (no insert/delete)

3. SETS:
   - Add/Remove: O(1) average
   - Search: O(1) average
   - Union/Intersection: O(n+m)

4. DICTIONARIES:
   - Access by key: O(1) average
   - Add/Remove: O(1) average
   - Search: O(1) average

Choose the right collection for your use case!
""")

# =============================================================================
# BEST PRACTICES
# =============================================================================

print("\n" + "="*60)
print("BEST PRACTICES")
print("="*60)

print("""
1. COLLECTION SELECTION:
   - Use lists for ordered, mutable sequences
   - Use tuples for fixed, immutable data
   - Use sets for unique elements and fast membership testing
   - Use dictionaries for key-value mappings

2. COMPREHENSIONS:
   - Prefer comprehensions over loops for simple transformations
   - Keep comprehensions readable - use multiple lines if needed
   - Use generator expressions for large datasets

3. STRING OPERATIONS:
   - Use f-strings for most formatting needs
   - Use join() for concatenating multiple strings
   - Be aware of string immutability

4. PERFORMANCE:
   - Choose appropriate data structures
   - Use built-in methods when possible
   - Consider memory usage for large datasets

5. READABILITY:
   - Use meaningful variable names
   - Add comments for complex operations
   - Follow PEP 8 style guidelines
""")

print("\n" + "="*60)
print("MODULE 2 COMPLETE!")
print("Next: Module 3 - Code Organization, Functions & Error Handling")
print("="*60)