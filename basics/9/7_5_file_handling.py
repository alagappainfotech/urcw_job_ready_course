# Lesson: Python File Handling
# File handling allows you to read, write, and manipulate files in Python.

# 1. Opening a File
# Use the `open()` function to open a file. Modes:
# 'r' - Read (default), 'w' - Write, 'a' - Append, 'x' - Create, 'b' - Binary mode.

# Example: Opening a file in read mode
try:
    file = open("example.txt", "r")
    print(file.read())  # Reads the entire file
    file.close()
except FileNotFoundError:
    print("Error: File not found.")

# 2. Writing to a File
# Use 'w' mode to write to a file. If the file exists, it will be overwritten.

with open("example.txt", "w") as file:
    file.write("Hello, world!\n")
    file.write("This is a new line.")

# 3. Appending to a File
# Use 'a' mode to append content to an existing file.

with open("example.txt", "a") as file:
    file.write("Appending this line.\n")

# 4. Reading a File Line by Line
# Use a loop to read a file line by line.

with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())  # Removes trailing newline characters

# 5. File Modes
# 'rb' - Read binary, 'wb' - Write binary, etc.

# Example: Writing and reading binary data
with open("binary_file.bin", "wb") as file:
    file.write(b"This is binary data.")

with open("binary_file.bin", "rb") as file:
    print(file.read())

# 6. Checking if a File Exists
# Use the `os` module to check if a file exists.

import os
if os.path.exists("example.txt"):
    print("File exists.")
else:
    print("File does not exist.")

# 7. Deleting a File
# Use the `os` module to delete a file.

if os.path.exists("example.txt"):
    os.remove("example.txt")
    print("File deleted.")
else:
    print("File does not exist.")

# Exercise:
# 1. Write a program to create a file and write the numbers 1 to 10, each on a new line.
# 2. Write a program to read a file and count the number of lines.
# 3. Write a program to copy the contents of one file to another.