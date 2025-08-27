####################################################
## Exercise 10: Regular Expressions (Regex)
####################################################

# Complete the following functions and test them

import re
from datetime import datetime

def is_valid_date(date_string):
    """
    Validate date format YYYY-MM-DD
    
    Args:
        date_string (str): Date string to validate
        
    Returns:
        bool: True if valid date format, False otherwise
        
    Example:
        >>> is_valid_date("2023-12-15")
        True
        >>> is_valid_date("2023-13-45")
        False
        >>> is_valid_date("invalid-date")
        False
    """
    # Your code here
    pass

def extract_urls(text):
    """
    Extract all URLs from a text
    
    Args:
        text (str): Text to search for URLs
        
    Returns:
        list: List of URLs found
        
    Example:
        >>> text = "Visit https://example.com and http://google.com"
        >>> extract_urls(text)
        ['https://example.com', 'http://google.com']
    """
    # Your code here
    pass

def is_valid_credit_card(card_number):
    """
    Validate credit card number format (basic validation)
    
    Args:
        card_number (str): Credit card number
        
    Returns:
        bool: True if valid format, False otherwise
        
    Example:
        >>> is_valid_credit_card("1234-5678-9012-3456")
        True
        >>> is_valid_credit_card("1234-5678-9012")
        False
    """
    # Your code here
    pass

def find_vowel_words(text):
    """
    Find all words that start with a vowel
    
    Args:
        text (str): Text to search
        
    Returns:
        list: List of words starting with vowels
        
    Example:
        >>> find_vowel_words("Apple orange banana grape")
        ['Apple', 'orange']
    """
    # Your code here
    pass

def validate_phone_number(phone):
    """
    Validate phone number format (various formats)
    
    Args:
        phone (str): Phone number to validate
        
    Returns:
        bool: True if valid format, False otherwise
        
    Example:
        >>> validate_phone_number("(123) 456-7890")
        True
        >>> validate_phone_number("123-456-7890")
        True
        >>> validate_phone_number("123.456.7890")
        True
        >>> validate_phone_number("1234567890")
        True
        >>> validate_phone_number("invalid")
        False
    """
    # Your code here
    pass

def extract_emails(text):
    """
    Extract all email addresses from text
    
    Args:
        text (str): Text to search for emails
        
    Returns:
        list: List of email addresses found
        
    Example:
        >>> text = "Contact john@example.com or jane@test.org"
        >>> extract_emails(text)
        ['john@example.com', 'jane@test.org']
    """
    # Your code here
    pass

def validate_password_strength(password):
    """
    Validate password strength using regex
    
    Requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character
    
    Args:
        password (str): Password to validate
        
    Returns:
        dict: Dictionary with validation results
        
    Example:
        >>> validate_password_strength("weak")
        {'valid': False, 'errors': ['Too short', 'No uppercase', 'No digit', 'No special char']}
        >>> validate_password_strength("StrongPass123!")
        {'valid': True, 'errors': []}
    """
    # Your code here
    pass

def extract_ip_addresses(text):
    """
    Extract IPv4 addresses from text
    
    Args:
        text (str): Text to search for IP addresses
        
    Returns:
        list: List of IP addresses found
        
    Example:
        >>> text = "Server IP: 192.168.1.1 and backup: 10.0.0.1"
        >>> extract_ip_addresses(text)
        ['192.168.1.1', '10.0.0.1']
    """
    # Your code here
    pass

def clean_text(text):
    """
    Clean text by removing extra whitespace and normalizing
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
        
    Example:
        >>> clean_text("  This   is    a    messy    text   ")
        'This is a messy text'
    """
    # Your code here
    pass

def extract_numbers(text):
    """
    Extract all numbers (integers and decimals) from text
    
    Args:
        text (str): Text to search for numbers
        
    Returns:
        list: List of numbers found (as strings)
        
    Example:
        >>> extract_numbers("The price is $19.99 and quantity is 5")
        ['19.99', '5']
    """
    # Your code here
    pass

def validate_time_format(time_string):
    """
    Validate time format HH:MM (24-hour format)
    
    Args:
        time_string (str): Time string to validate
        
    Returns:
        bool: True if valid time format, False otherwise
        
    Example:
        >>> validate_time_format("14:30")
        True
        >>> validate_time_format("25:70")
        False
        >>> validate_time_format("9:5")
        False
    """
    # Your code here
    pass

def extract_hashtags(text):
    """
    Extract hashtags from text
    
    Args:
        text (str): Text to search for hashtags
        
    Returns:
        list: List of hashtags found
        
    Example:
        >>> extract_hashtags("Check out #python #programming #coding")
        ['#python', '#programming', '#coding']
    """
    # Your code here
    pass

def validate_postal_code(postal_code):
    """
    Validate US postal code format (5 digits or 5+4 format)
    
    Args:
        postal_code (str): Postal code to validate
        
    Returns:
        bool: True if valid format, False otherwise
        
    Example:
        >>> validate_postal_code("12345")
        True
        >>> validate_postal_code("12345-6789")
        True
        >>> validate_postal_code("1234")
        False
    """
    # Your code here
    pass

def extract_quoted_text(text):
    """
    Extract text within quotes (single or double)
    
    Args:
        text (str): Text to search for quoted content
        
    Returns:
        list: List of quoted text found
        
    Example:
        >>> extract_quoted_text('He said "Hello" and she replied \'Hi\'')
        ['Hello', 'Hi']
    """
    # Your code here
    pass

def validate_ssn(ssn):
    """
    Validate US Social Security Number format (XXX-XX-XXXX)
    
    Args:
        ssn (str): SSN to validate
        
    Returns:
        bool: True if valid format, False otherwise
        
    Example:
        >>> validate_ssn("123-45-6789")
        True
        >>> validate_ssn("123456789")
        False
    """
    # Your code here
    pass

def extract_file_extensions(text):
    """
    Extract file extensions from text
    
    Args:
        text (str): Text to search for file extensions
        
    Returns:
        list: List of file extensions found
        
    Example:
        >>> extract_file_extensions("Files: document.pdf, image.jpg, script.py")
        ['.pdf', '.jpg', '.py']
    """
    # Your code here
    pass

# Test your functions
if __name__ == "__main__":
    # Test is_valid_date
    print("Testing is_valid_date:")
    print(is_valid_date("2023-12-15"))  # Expected: True
    print(is_valid_date("2023-13-45"))  # Expected: False
    print(is_valid_date("invalid-date"))  # Expected: False
    print()
    
    # Test extract_urls
    print("Testing extract_urls:")
    text = "Visit https://example.com and http://google.com for more info"
    print(extract_urls(text))  # Expected: ['https://example.com', 'http://google.com']
    print()
    
    # Test is_valid_credit_card
    print("Testing is_valid_credit_card:")
    print(is_valid_credit_card("1234-5678-9012-3456"))  # Expected: True
    print(is_valid_credit_card("1234-5678-9012"))  # Expected: False
    print()
    
    # Test find_vowel_words
    print("Testing find_vowel_words:")
    text = "Apple orange banana grape cherry"
    print(find_vowel_words(text))  # Expected: ['Apple', 'orange']
    print()
    
    # Test validate_phone_number
    print("Testing validate_phone_number:")
    print(validate_phone_number("(123) 456-7890"))  # Expected: True
    print(validate_phone_number("123-456-7890"))  # Expected: True
    print(validate_phone_number("invalid"))  # Expected: False
    print()
    
    # Test extract_emails
    print("Testing extract_emails:")
    text = "Contact john@example.com or jane@test.org for support"
    print(extract_emails(text))  # Expected: ['john@example.com', 'jane@test.org']
    print()
    
    # Test validate_password_strength
    print("Testing validate_password_strength:")
    print(validate_password_strength("weak"))  # Expected: {'valid': False, 'errors': [...]}
    print(validate_password_strength("StrongPass123!"))  # Expected: {'valid': True, 'errors': []}
    print()
    
    # Test extract_ip_addresses
    print("Testing extract_ip_addresses:")
    text = "Server IP: 192.168.1.1 and backup: 10.0.0.1"
    print(extract_ip_addresses(text))  # Expected: ['192.168.1.1', '10.0.0.1']
    print()
    
    # Test clean_text
    print("Testing clean_text:")
    text = "  This   is    a    messy    text   "
    print(clean_text(text))  # Expected: 'This is a messy text'
    print()
    
    # Test extract_numbers
    print("Testing extract_numbers:")
    text = "The price is $19.99 and quantity is 5"
    print(extract_numbers(text))  # Expected: ['19.99', '5']
    print()
    
    # Test validate_time_format
    print("Testing validate_time_format:")
    print(validate_time_format("14:30"))  # Expected: True
    print(validate_time_format("25:70"))  # Expected: False
    print()
    
    # Test extract_hashtags
    print("Testing extract_hashtags:")
    text = "Check out #python #programming #coding"
    print(extract_hashtags(text))  # Expected: ['#python', '#programming', '#coding']
    print()
    
    # Test validate_postal_code
    print("Testing validate_postal_code:")
    print(validate_postal_code("12345"))  # Expected: True
    print(validate_postal_code("12345-6789"))  # Expected: True
    print(validate_postal_code("1234"))  # Expected: False
    print()
    
    # Test extract_quoted_text
    print("Testing extract_quoted_text:")
    text = 'He said "Hello" and she replied \'Hi\''
    print(extract_quoted_text(text))  # Expected: ['Hello', 'Hi']
    print()
    
    # Test validate_ssn
    print("Testing validate_ssn:")
    print(validate_ssn("123-45-6789"))  # Expected: True
    print(validate_ssn("123456789"))  # Expected: False
    print()
    
    # Test extract_file_extensions
    print("Testing extract_file_extensions:")
    text = "Files: document.pdf, image.jpg, script.py"
    print(extract_file_extensions(text))  # Expected: ['.pdf', '.jpg', '.py']
    print() 