# Lesson: Python Data Structures
# Python provides several built-in data structures to store and manage data efficiently.

# 1. Lists
# Lists are ordered, mutable collections of items.
fruits = ['apple', 'banana', 'cherry']
print(fruits)  # Output: ['apple', 'banana', 'cherry']

# Adding elements
fruits.append('orange')
print(fruits)  # Output: ['apple', 'banana', 'cherry', 'orange']

# Removing elements
fruits.remove('banana')
print(fruits)  # Output: ['apple', 'cherry', 'orange']

# Accessing elements
print(fruits[0])  # Output: apple

# 2. Tuples
# Tuples are ordered, immutable collections of items.
coordinates = (10, 20)
print(coordinates)  # Output: (10, 20)

# Accessing elements
print(coordinates[1])  # Output: 20

# 3. Sets
# Sets are unordered collections of unique items.
unique_numbers = {1, 2, 3, 3}
print(unique_numbers)  # Output: {1, 2, 3}

# Adding elements
unique_numbers.add(4)
print(unique_numbers)  # Output: {1, 2, 3, 4}

# Removing elements
unique_numbers.remove(2)
print(unique_numbers)  # Output: {1, 3, 4}

# 4. Dictionaries
# Dictionaries store key-value pairs.
person = {'name': 'Alice', 'age': 25}
print(person)  # Output: {'name': 'Alice', 'age': 25}

# Accessing values
print(person['name'])  # Output: Alice

# Adding key-value pairs
person['city'] = 'New York'
print(person)  # Output: {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Removing key-value pairs
del person['age']
print(person)  # Output: {'name': 'Alice', 'city': 'New York'}

# 5. Strings as Data Structures
# Strings are immutable sequences of characters.
text = "hello"
print(text[1])  # Output: e

# Slicing
print(text[1:4])  # Output: ell

# 6. Nested Data Structures
# Data structures can be nested to create complex structures.
nested_list = [[1, 2], [3, 4]]
print(nested_list[1][0])  # Output: 3

nested_dict = {'person': {'name': 'Bob', 'age': 30}}
print(nested_dict['person']['name'])  # Output: Bob

# Exercise:
# 1. Create a list of dictionaries to store information about three students (name, age, grade).
# 2. Write a program to find the union and intersection of two sets.
# 3. Create a nested dictionary to represent a company's departments and employees.