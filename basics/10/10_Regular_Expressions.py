####################################################
## 10. Regular Expressions (Regex)
####################################################

# Regular expressions are patterns used to match character combinations in strings
# Python's re module provides support for regular expressions

import re

# 1. Basic Pattern Matching

# Simple string matching
text = "Hello, World!"
pattern = "Hello"
match = re.search(pattern, text)
if match:
    print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")

# Case-insensitive matching
pattern = re.compile(r"hello", re.IGNORECASE)
match = pattern.search("HELLO, World!")
if match:
    print(f"Found '{match.group()}' (case-insensitive)")

# 2. Character Classes and Special Characters

# Character classes
text = "The quick brown fox jumps over the lazy dog"
pattern = r"[aeiou]"  # Match any vowel
matches = re.findall(pattern, text)
print(f"Vowels found: {matches}")

# Negated character class
pattern = r"[^aeiou\s]"  # Match any non-vowel, non-whitespace
matches = re.findall(pattern, text)
print(f"Consonants found: {matches}")

# Predefined character classes
text = "My phone number is 123-456-7890"
pattern = r"\d"  # Match any digit
digits = re.findall(pattern, text)
print(f"Digits found: {digits}")

pattern = r"\w"  # Match any word character (letter, digit, underscore)
word_chars = re.findall(pattern, text)
print(f"Word characters found: {word_chars}")

pattern = r"\s"  # Match any whitespace character
spaces = re.findall(pattern, text)
print(f"Whitespace characters found: {spaces}")

# 3. Quantifiers

# Zero or more (*)
text = "aaabbbccc"
pattern = r"a*"  # Match zero or more 'a's
matches = re.findall(pattern, text)
print(f"Zero or more 'a's: {matches}")

# One or more (+)
pattern = r"a+"  # Match one or more 'a's
matches = re.findall(pattern, text)
print(f"One or more 'a's: {matches}")

# Zero or one (?)
text = "color colour"
pattern = r"colou?r"  # Match 'color' or 'colour'
matches = re.findall(pattern, text)
print(f"Color/colour matches: {matches}")

# Specific number of repetitions
text = "aaa aaaa aaaaa"
pattern = r"a{3}"  # Match exactly 3 'a's
matches = re.findall(pattern, text)
print(f"Exactly 3 'a's: {matches}")

pattern = r"a{3,5}"  # Match 3 to 5 'a's
matches = re.findall(pattern, text)
print(f"3 to 5 'a's: {matches}")

# 4. Anchors and Boundaries

text = "Start of line\nMiddle of line\nEnd of line"

# Start of line (^)
pattern = r"^Start"
matches = re.findall(pattern, text, re.MULTILINE)
print(f"Lines starting with 'Start': {matches}")

# End of line ($)
pattern = r"line$"
matches = re.findall(pattern, text, re.MULTILINE)
print(f"Lines ending with 'line': {matches}")

# Word boundaries (\b)
text = "The word 'the' appears multiple times in this sentence."
pattern = r"\bthe\b"  # Match 'the' as a whole word
matches = re.findall(pattern, text, re.IGNORECASE)
print(f"Whole word 'the': {matches}")

# 5. Groups and Capturing

# Simple groups
text = "John Doe, Jane Smith, Bob Johnson"
pattern = r"(\w+)\s+(\w+)"  # Capture first and last names
matches = re.findall(pattern, text)
print(f"Name pairs: {matches}")

# Named groups
pattern = r"(?P<first>\w+)\s+(?P<last>\w+)"
match = re.search(pattern, text)
if match:
    print(f"First name: {match.group('first')}")
    print(f"Last name: {match.group('last')}")

# Non-capturing groups
text = "color colour"
pattern = r"colou(?:r)?"  # Non-capturing group
matches = re.findall(pattern, text)
print(f"Color matches (non-capturing): {matches}")

# 6. Alternation and Lookahead/Lookbehind

# Alternation (|)
text = "cat dog bird fish"
pattern = r"cat|dog"  # Match 'cat' or 'dog'
matches = re.findall(pattern, text)
print(f"Cat or dog: {matches}")

# Positive lookahead
text = "apple123 banana456 cherry789"
pattern = r"\w+(?=\d)"  # Match word followed by digit
matches = re.findall(pattern, text)
print(f"Words followed by digits: {matches}")

# Negative lookahead
pattern = r"\w+(?!\d)"  # Match word not followed by digit
matches = re.findall(pattern, text)
print(f"Words not followed by digits: {matches}")

# 7. Substitution and Replacement

# Simple substitution
text = "Hello, World! Hello, Python!"
pattern = r"Hello"
replacement = "Hi"
new_text = re.sub(pattern, replacement, text)
print(f"Substitution: {new_text}")

# Substitution with groups
text = "John Doe, Jane Smith"
pattern = r"(\w+)\s+(\w+)"
replacement = r"\2, \1"  # Last name, First name
new_text = re.sub(pattern, replacement, text)
print(f"Name format change: {new_text}")

# 8. Practical Examples

# Email validation
def is_valid_email(email):
    """Validate email format using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

emails = ["user@example.com", "invalid-email", "test@domain.co.uk"]
for email in emails:
    print(f"{email}: {'Valid' if is_valid_email(email) else 'Invalid'}")

# Phone number formatting
def format_phone_number(phone):
    """Format phone number to (XXX) XXX-XXXX"""
    pattern = r'(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})'
    replacement = r'(\1) \2-\3'
    return re.sub(pattern, replacement, phone)

phone_numbers = ["1234567890", "123-456-7890", "123.456.7890"]
for phone in phone_numbers:
    print(f"{phone} -> {format_phone_number(phone)}")

# Password strength checker
def check_password_strength(password):
    """Check password strength using regex"""
    checks = {
        'length': len(password) >= 8,
        'lowercase': bool(re.search(r'[a-z]', password)),
        'uppercase': bool(re.search(r'[A-Z]', password)),
        'digit': bool(re.search(r'\d', password)),
        'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    
    score = sum(checks.values())
    if score == 5:
        return "Very Strong"
    elif score >= 4:
        return "Strong"
    elif score >= 3:
        return "Moderate"
    else:
        return "Weak"

passwords = ["abc123", "Password123", "Str0ng!P@ss"]
for pwd in passwords:
    print(f"'{pwd}': {check_password_strength(pwd)}")

# URL parsing
def parse_url(url):
    """Parse URL components using regex"""
    pattern = r'^(https?://)?([^/]+)(/.*)?$'
    match = re.match(pattern, url)
    if match:
        protocol = match.group(1) or 'http://'
        domain = match.group(2)
        path = match.group(3) or '/'
        return {
            'protocol': protocol,
            'domain': domain,
            'path': path
        }
    return None

urls = ["https://example.com/path", "http://google.com", "ftp://server.com"]
for url in urls:
    parsed = parse_url(url)
    print(f"{url} -> {parsed}")

# 9. Advanced Patterns

# Matching nested parentheses
def match_nested_parens(text):
    """Match content within nested parentheses"""
    pattern = r'\(([^()]*|\([^()]*\))*\)'
    matches = re.findall(pattern, text)
    return matches

text = "This (has (nested) parentheses) and (another set)"
print(f"Nested parentheses: {match_nested_parens(text)}")

# Extracting data from log files
def parse_log_line(log_line):
    """Parse a log line with timestamp, level, and message"""
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
    match = re.match(pattern, log_line)
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2),
            'message': match.group(3)
        }
    return None

log_lines = [
    "2023-01-15 10:30:45 [INFO] User login successful",
    "2023-01-15 10:31:12 [ERROR] Database connection failed"
]

for line in log_lines:
    parsed = parse_log_line(line)
    print(f"Log entry: {parsed}")

# 10. Exercises

# Exercise 1: Create a function that validates a date format (YYYY-MM-DD)
def is_valid_date(date_string):
    """Validate date format YYYY-MM-DD"""
    # Your code here
    pass

# Exercise 2: Create a function that extracts all URLs from a text
def extract_urls(text):
    """Extract all URLs from a text"""
    # Your code here
    pass

# Exercise 3: Create a function that validates a credit card number
def is_valid_credit_card(card_number):
    """Validate credit card number format"""
    # Your code here
    pass

# Exercise 4: Create a function that finds all words that start with a vowel
def find_vowel_words(text):
    """Find all words that start with a vowel"""
    # Your code here
    pass

# Test your functions here 