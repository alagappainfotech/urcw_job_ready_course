####################################################
## 8. String Manipulation and Formatting
####################################################

# Strings are immutable sequences of characters
# Python provides many methods for string manipulation

# 1. String Creation and Basic Operations
text = "Hello, World!"
print(text)  # Output: Hello, World!

# String concatenation
first = "Hello"
second = "World"
result = first + ", " + second + "!"
print(result)  # Output: Hello, World!

# String repetition
dashes = "-" * 20
print(dashes)  # Output: --------------------

# 2. String Methods
text = "  hello world  "
print(text.upper())      # Output: HELLO WORLD
print(text.lower())      # Output: hello world
print(text.title())      # Output: Hello World
print(text.strip())      # Output: hello world
print(text.lstrip())     # Output: hello world  
print(text.rstrip())     # Output:   hello world

# String replacement
text = "Hello World"
new_text = text.replace("World", "Python")
print(new_text)  # Output: Hello Python

# String splitting and joining
text = "apple,banana,cherry"
fruits = text.split(",")
print(fruits)  # Output: ['apple', 'banana', 'cherry']

fruits_text = " and ".join(fruits)
print(fruits_text)  # Output: apple and banana and cherry

# 3. String Formatting Methods

# Old-style formatting (%)
name = "Alice"
age = 30
print("My name is %s and I am %d years old" % (name, age))

# str.format() method
print("My name is {} and I am {} years old".format(name, age))
print("My name is {0} and I am {1} years old".format(name, age))
print("My name is {n} and I am {a} years old".format(n=name, a=age))

# f-strings (Python 3.6+)
print(f"My name is {name} and I am {age} years old")

# Formatting with precision
pi = 3.14159
print(f"Pi to 2 decimal places: {pi:.2f}")  # Output: Pi to 2 decimal places: 3.14

# 4. String Slicing and Indexing
text = "Python Programming"
print(text[0])      # Output: P
print(text[-1])     # Output: g
print(text[0:6])    # Output: Python
print(text[7:])     # Output: Programming
print(text[:6])     # Output: Python
print(text[::2])    # Output: Pto rgamn

# 5. String Testing Methods
text = "Hello123"
print(text.isalpha())    # False (contains numbers)
print(text.isalnum())    # True (alphanumeric)
print(text.isdigit())    # False
print(text.startswith("Hello"))  # True
print(text.endswith("123"))      # True

# 6. String Search and Count
text = "hello world hello python"
print(text.count("hello"))  # Output: 2
print(text.find("world"))   # Output: 6
print(text.find("python"))  # Output: 18
print(text.find("java"))    # Output: -1 (not found)

# 7. Advanced String Operations
import string

# String constants
print(string.ascii_letters)  # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.digits)         # 0123456789
print(string.punctuation)    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# String translation
text = "Hello World!"
translation_table = str.maketrans("aeiou", "12345")
print(text.translate(translation_table))  # Output: H2ll4 W4rld!

# 8. Multiline Strings
multiline = """
This is a multiline string.
It can span multiple lines.
Useful for documentation and long text.
"""
print(multiline)

# Raw strings (for regex patterns, file paths)
raw_string = r"C:\Users\Name\Documents"
print(raw_string)  # Output: C:\Users\Name\Documents

# 9. String Templates
from string import Template

template = Template("Hello, $name! You are $age years old.")
result = template.substitute(name="Bob", age=25)
print(result)  # Output: Hello, Bob! You are 25 years old.

# 10. Practical Examples

# Email validation (basic)
def is_valid_email(email):
    return "@" in email and "." in email and email.count("@") == 1

print(is_valid_email("user@example.com"))  # True
print(is_valid_email("invalid-email"))     # False

# Password strength checker
def check_password_strength(password):
    if len(password) < 8:
        return "Weak: Too short"
    elif not any(c.isupper() for c in password):
        return "Weak: No uppercase letters"
    elif not any(c.islower() for c in password):
        return "Weak: No lowercase letters"
    elif not any(c.isdigit() for c in password):
        return "Weak: No numbers"
    else:
        return "Strong password"

print(check_password_strength("abc123"))      # Weak: Too short
print(check_password_strength("Password123")) # Strong password

# Text processing
def clean_text(text):
    """Remove extra whitespace and normalize text"""
    return " ".join(text.split())

text = "  This   is    a    messy    text   "
print(clean_text(text))  # Output: This is a messy text

# Exercise 1: Create a function that capitalizes the first letter of each word in a sentence
def title_case(sentence):
    """Convert a sentence to title case"""
    # Your code here
    pass

# Exercise 2: Create a function that checks if a string is a palindrome
def is_palindrome(text):
    """Check if a string is a palindrome (reads the same forwards and backwards)"""
    # Your code here
    pass

# Exercise 3: Create a function that counts vowels and consonants in a string
def count_letters(text):
    """Count vowels and consonants in a string"""
    # Your code here
    pass

# Exercise 4: Create a function that formats a phone number
def format_phone_number(phone):
    """Format a 10-digit phone number as (XXX) XXX-XXXX"""
    # Your code here
    pass

# Test your functions here 