####################################################
## 9. File Handling and I/O Operations
####################################################

# Python provides built-in functions and methods for file operations
# Files can be opened in different modes: read, write, append, etc.

# 1. Basic File Operations

# Opening and reading a file
with open('sample.txt', 'w') as file:
    file.write("Hello, this is a sample file.\n")
    file.write("This is the second line.\n")
    file.write("And this is the third line.")

# Reading the entire file
with open('sample.txt', 'r') as file:
    content = file.read()
    print("Full content:")
    print(content)

# Reading line by line
with open('sample.txt', 'r') as file:
    print("\nLine by line:")
    for line in file:
        print(f"Line: {line.strip()}")

# Reading all lines into a list
with open('sample.txt', 'r') as file:
    lines = file.readlines()
    print(f"\nAll lines as list: {lines}")

# 2. File Modes
"""
'r'  - Read (default)
'w'  - Write (overwrites existing content)
'a'  - Append (adds to existing content)
'x'  - Exclusive creation (fails if file exists)
'b'  - Binary mode
't'  - Text mode (default)
'+'  - Read and write
"""

# Append mode example
with open('sample.txt', 'a') as file:
    file.write("\nThis line was appended.")

# 3. Working with CSV Files
import csv

# Writing CSV data
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Los Angeles'],
    ['Charlie', 35, 'Chicago']
]

with open('people.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Reading CSV data
with open('people.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"CSV Row: {row}")

# 4. Working with JSON Files
import json

# Writing JSON data
data = {
    'name': 'Alice',
    'age': 25,
    'city': 'New York',
    'hobbies': ['reading', 'swimming', 'coding']
}

with open('person.json', 'w') as file:
    json.dump(data, file, indent=2)

# Reading JSON data
with open('person.json', 'r') as file:
    loaded_data = json.load(file)
    print(f"\nJSON data: {loaded_data}")

# 5. Working with Binary Files
# Writing binary data
binary_data = b'\x48\x65\x6c\x6c\x6f\x20\x57\x6f\x72\x6c\x64'  # "Hello World"
with open('binary_file.bin', 'wb') as file:
    file.write(binary_data)

# Reading binary data
with open('binary_file.bin', 'rb') as file:
    binary_content = file.read()
    print(f"\nBinary content: {binary_content}")
    print(f"As text: {binary_content.decode('utf-8')}")

# 6. File and Directory Operations
import os
import shutil

# Check if file exists
print(f"\nFile exists: {os.path.exists('sample.txt')}")

# Get file information
file_stats = os.stat('sample.txt')
print(f"File size: {file_stats.st_size} bytes")
print(f"Last modified: {file_stats.st_mtime}")

# List directory contents
print(f"\nCurrent directory contents:")
for item in os.listdir('.'):
    if os.path.isfile(item):
        print(f"File: {item}")
    elif os.path.isdir(item):
        print(f"Directory: {item}")

# 7. Working with Paths
from pathlib import Path

# Create a Path object
path = Path('sample.txt')
print(f"\nFile name: {path.name}")
print(f"File suffix: {path.suffix}")
print(f"File stem: {path.stem}")
print(f"Parent directory: {path.parent}")

# Create directories
Path('new_folder').mkdir(exist_ok=True)

# 8. Error Handling in File Operations
def safe_read_file(filename):
    """Safely read a file with proper error handling"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test the safe read function
content = safe_read_file('nonexistent.txt')
print(f"\nSafe read result: {content}")

# 9. Context Managers and Custom File Handlers
class FileLogger:
    """Custom context manager for logging file operations"""
    
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            print(f"Closing file: {self.filename}")
            self.file.close()
        if exc_type:
            print(f"Error occurred: {exc_type}")
        return False  # Don't suppress exceptions

# Using custom context manager
with FileLogger('sample.txt', 'r') as file:
    content = file.read()
    print(f"Content length: {len(content)} characters")

# 10. Practical Examples

# Log file processor
def process_log_file(log_filename):
    """Process a log file and count different log levels"""
    log_counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'DEBUG': 0}
    
    try:
        with open(log_filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    for level in log_counts.keys():
                        if level in line:
                            log_counts[level] += 1
                            break
    except FileNotFoundError:
        print(f"Log file '{log_filename}' not found.")
        return None
    
    return log_counts

# Configuration file reader
def read_config(config_filename):
    """Read a simple configuration file"""
    config = {}
    
    try:
        with open(config_filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Config file '{config_filename}' not found.")
        return None
    
    return config

# File backup utility
def backup_file(filename):
    """Create a backup of a file"""
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return False
    
    backup_name = f"{filename}.backup"
    try:
        shutil.copy2(filename, backup_name)
        print(f"Backup created: {backup_name}")
        return True
    except Exception as e:
        print(f"Backup failed: {e}")
        return False

# 11. Exercises

# Exercise 1: Create a function that counts words in a text file
def count_words_in_file(filename):
    """Count the number of words in a text file"""
    # Your code here
    pass

# Exercise 2: Create a function that finds the longest line in a file
def find_longest_line(filename):
    """Find the longest line in a text file"""
    # Your code here
    pass

# Exercise 3: Create a function that creates a simple log file
def write_log_entry(message, level="INFO"):
    """Write a log entry to a log file"""
    # Your code here
    pass

# Exercise 4: Create a function that merges two text files
def merge_files(file1, file2, output_file):
    """Merge the contents of two text files into a third file"""
    # Your code here
    pass

# Test your functions here 