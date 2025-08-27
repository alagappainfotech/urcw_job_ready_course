####################################################
## Exercise 9: File Handling and I/O Operations
####################################################

# Complete the following functions and test them

import os
import json
import csv
from datetime import datetime

def count_words_in_file(filename):
    """
    Count the number of words in a text file
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        int: Number of words in the file
        
    Example:
        >>> count_words_in_file("sample.txt")
        15
    """
    # Your code here
    pass

def find_longest_line(filename):
    """
    Find the longest line in a text file
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        tuple: (line_number, line_content, line_length)
        
    Example:
        >>> find_longest_line("sample.txt")
        (3, "This is the longest line in the file", 35)
    """
    # Your code here
    pass

def write_log_entry(message, level="INFO", filename="app.log"):
    """
    Write a log entry to a log file with timestamp
    
    Args:
        message (str): The log message
        level (str): Log level (INFO, WARNING, ERROR, DEBUG)
        filename (str): Log file name
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> write_log_entry("User login successful", "INFO")
        True
    """
    # Your code here
    pass

def merge_files(file1, file2, output_file):
    """
    Merge the contents of two text files into a third file
    
    Args:
        file1 (str): Path to first file
        file2 (str): Path to second file
        output_file (str): Path to output file
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> merge_files("file1.txt", "file2.txt", "merged.txt")
        True
    """
    # Your code here
    pass

def create_csv_from_dict(data, filename):
    """
    Create a CSV file from a list of dictionaries
    
    Args:
        data (list): List of dictionaries
        filename (str): Output CSV filename
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> data = [
        ...     {'name': 'John', 'age': 30, 'city': 'NYC'},
        ...     {'name': 'Jane', 'age': 25, 'city': 'LA'}
        ... ]
        >>> create_csv_from_dict(data, "people.csv")
        True
    """
    # Your code here
    pass

def read_csv_to_dict(filename):
    """
    Read a CSV file and return as a list of dictionaries
    
    Args:
        filename (str): Path to CSV file
        
    Returns:
        list: List of dictionaries
        
    Example:
        >>> read_csv_to_dict("people.csv")
        [{'name': 'John', 'age': '30', 'city': 'NYC'}, ...]
    """
    # Your code here
    pass

def backup_file(filename):
    """
    Create a backup of a file with timestamp
    
    Args:
        filename (str): Path to file to backup
        
    Returns:
        str: Path to backup file, or None if failed
        
    Example:
        >>> backup_file("important.txt")
        'important.txt.backup.20231215_143022'
    """
    # Your code here
    pass

def find_files_by_extension(directory, extension):
    """
    Find all files with a specific extension in a directory
    
    Args:
        directory (str): Directory to search
        extension (str): File extension (without dot)
        
    Returns:
        list: List of file paths
        
    Example:
        >>> find_files_by_extension(".", "txt")
        ['file1.txt', 'file2.txt', 'notes.txt']
    """
    # Your code here
    pass

def create_config_file(config_data, filename="config.ini"):
    """
    Create a configuration file from a dictionary
    
    Args:
        config_data (dict): Configuration data
        filename (str): Output filename
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> config = {
        ...     'database': {'host': 'localhost', 'port': 5432},
        ...     'app': {'debug': True, 'secret_key': 'abc123'}
        ... }
        >>> create_config_file(config, "app.ini")
        True
    """
    # Your code here
    pass

def read_config_file(filename="config.ini"):
    """
    Read a configuration file and return as dictionary
    
    Args:
        filename (str): Configuration file path
        
    Returns:
        dict: Configuration data
        
    Example:
        >>> read_config_file("app.ini")
        {'database': {'host': 'localhost', 'port': '5432'}, ...}
    """
    # Your code here
    pass

def process_log_file(log_filename):
    """
    Process a log file and count different log levels
    
    Args:
        log_filename (str): Path to log file
        
    Returns:
        dict: Dictionary with log level counts
        
    Example:
        >>> process_log_file("app.log")
        {'INFO': 10, 'WARNING': 3, 'ERROR': 1, 'DEBUG': 5}
    """
    # Your code here
    pass

def create_json_backup(data, filename):
    """
    Create a JSON backup of data with timestamp
    
    Args:
        data (dict): Data to backup
        filename (str): Base filename
        
    Returns:
        str: Path to backup file
        
    Example:
        >>> data = {'users': [{'name': 'John'}, {'name': 'Jane'}]}
        >>> create_json_backup(data, "users.json")
        'users.backup.20231215_143022.json'
    """
    # Your code here
    pass

def validate_file_exists(filename):
    """
    Check if a file exists and is readable
    
    Args:
        filename (str): Path to file
        
    Returns:
        dict: Dictionary with validation results
        
    Example:
        >>> validate_file_exists("nonexistent.txt")
        {'exists': False, 'readable': False, 'size': 0}
        >>> validate_file_exists("existing.txt")
        {'exists': True, 'readable': True, 'size': 1024}
    """
    # Your code here
    pass

# Test your functions
if __name__ == "__main__":
    # Create test files
    with open("test_file1.txt", "w") as f:
        f.write("This is the first test file.\nIt has multiple lines.\nThis is the longest line in this file for testing purposes.")
    
    with open("test_file2.txt", "w") as f:
        f.write("This is the second test file.\nIt contains different content.\nShort line.\nAnother line with some content.")
    
    # Test count_words_in_file
    print("Testing count_words_in_file:")
    print(count_words_in_file("test_file1.txt"))  # Expected: ~20 words
    print()
    
    # Test find_longest_line
    print("Testing find_longest_line:")
    print(find_longest_line("test_file1.txt"))  # Expected: (3, "This is the longest line...", length)
    print()
    
    # Test write_log_entry
    print("Testing write_log_entry:")
    print(write_log_entry("Test log message", "INFO"))  # Expected: True
    print()
    
    # Test merge_files
    print("Testing merge_files:")
    print(merge_files("test_file1.txt", "test_file2.txt", "merged.txt"))  # Expected: True
    print()
    
    # Test create_csv_from_dict
    print("Testing create_csv_from_dict:")
    test_data = [
        {'name': 'John', 'age': 30, 'city': 'NYC'},
        {'name': 'Jane', 'age': 25, 'city': 'LA'},
        {'name': 'Bob', 'age': 35, 'city': 'Chicago'}
    ]
    print(create_csv_from_dict(test_data, "test_people.csv"))  # Expected: True
    print()
    
    # Test read_csv_to_dict
    print("Testing read_csv_to_dict:")
    print(read_csv_to_dict("test_people.csv"))  # Expected: List of dictionaries
    print()
    
    # Test backup_file
    print("Testing backup_file:")
    print(backup_file("test_file1.txt"))  # Expected: Backup filename
    print()
    
    # Test find_files_by_extension
    print("Testing find_files_by_extension:")
    print(find_files_by_extension(".", "txt"))  # Expected: List of .txt files
    print()
    
    # Test create_config_file
    print("Testing create_config_file:")
    test_config = {
        'database': {'host': 'localhost', 'port': 5432},
        'app': {'debug': True, 'secret_key': 'abc123'}
    }
    print(create_config_file(test_config, "test_config.ini"))  # Expected: True
    print()
    
    # Test read_config_file
    print("Testing read_config_file:")
    print(read_config_file("test_config.ini"))  # Expected: Dictionary
    print()
    
    # Test process_log_file
    print("Testing process_log_file:")
    # Create a test log file
    with open("test_app.log", "w") as f:
        f.write("2023-12-15 10:00:00 [INFO] Application started\n")
        f.write("2023-12-15 10:01:00 [WARNING] High memory usage\n")
        f.write("2023-12-15 10:02:00 [ERROR] Database connection failed\n")
        f.write("2023-12-15 10:03:00 [INFO] User login successful\n")
    print(process_log_file("test_app.log"))  # Expected: Dictionary with counts
    print()
    
    # Test create_json_backup
    print("Testing create_json_backup:")
    test_data = {'users': [{'name': 'John'}, {'name': 'Jane'}]}
    print(create_json_backup(test_data, "test_users.json"))  # Expected: Backup filename
    print()
    
    # Test validate_file_exists
    print("Testing validate_file_exists:")
    print(validate_file_exists("test_file1.txt"))  # Expected: {'exists': True, ...}
    print(validate_file_exists("nonexistent.txt"))  # Expected: {'exists': False, ...}
    print()
    
    # Clean up test files
    test_files = [
        "test_file1.txt", "test_file2.txt", "merged.txt", 
        "test_people.csv", "test_config.ini", "test_app.log",
        "test_users.backup.*.json"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file) 