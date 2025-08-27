####################################################
## Exercise 8: String Manipulation
####################################################

# Complete the following functions and test them

def title_case(sentence):
    """
    Convert a sentence to title case (capitalize first letter of each word)
    
    Args:
        sentence (str): The sentence to convert
        
    Returns:
        str: The sentence in title case
        
    Example:
        >>> title_case("hello world")
        'Hello World'
        >>> title_case("python programming language")
        'Python Programming Language'
    """
    # Your code here
    pass

def is_palindrome(text):
    """
    Check if a string is a palindrome (reads the same forwards and backwards)
    Ignore case and non-alphanumeric characters
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if palindrome, False otherwise
        
    Example:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("A man a plan a canal Panama")
        True
        >>> is_palindrome("hello")
        False
    """
    # Your code here
    pass

def count_letters(text):
    """
    Count vowels and consonants in a string
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with 'vowels' and 'consonants' counts
        
    Example:
        >>> count_letters("hello world")
        {'vowels': 3, 'consonants': 7}
        >>> count_letters("PYTHON")
        {'vowels': 1, 'consonants': 5}
    """
    # Your code here
    pass

def format_phone_number(phone):
    """
    Format a 10-digit phone number as (XXX) XXX-XXXX
    
    Args:
        phone (str): Phone number (can contain digits, spaces, dashes, dots)
        
    Returns:
        str: Formatted phone number
        
    Example:
        >>> format_phone_number("1234567890")
        '(123) 456-7890'
        >>> format_phone_number("123-456-7890")
        '(123) 456-7890'
        >>> format_phone_number("123.456.7890")
        '(123) 456-7890'
    """
    # Your code here
    pass

def extract_words(text):
    """
    Extract all words from a text, removing punctuation and converting to lowercase
    
    Args:
        text (str): The text to process
        
    Returns:
        list: List of words
        
    Example:
        >>> extract_words("Hello, World! How are you?")
        ['hello', 'world', 'how', 'are', 'you']
    """
    # Your code here
    pass

def word_frequency(text):
    """
    Count the frequency of each word in a text
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with words as keys and frequencies as values
        
    Example:
        >>> word_frequency("hello world hello python")
        {'hello': 2, 'world': 1, 'python': 1}
    """
    # Your code here
    pass

def validate_password(password):
    """
    Validate password strength
    
    Requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character (!@#$%^&*)
    
    Args:
        password (str): The password to validate
        
    Returns:
        dict: Dictionary with validation results
        
    Example:
        >>> validate_password("abc123")
        {'valid': False, 'errors': ['Too short', 'No uppercase', 'No special character']}
        >>> validate_password("Password123!")
        {'valid': True, 'errors': []}
    """
    # Your code here
    pass

def caesar_cipher(text, shift):
    """
    Implement Caesar cipher encryption/decryption
    
    Args:
        text (str): The text to encrypt/decrypt
        shift (int): The shift amount (positive for encryption, negative for decryption)
        
    Returns:
        str: The encrypted/decrypted text
        
    Example:
        >>> caesar_cipher("hello", 3)
        'khoor'
        >>> caesar_cipher("khoor", -3)
        'hello'
    """
    # Your code here
    pass

# Test your functions
if __name__ == "__main__":
    # Test title_case
    print("Testing title_case:")
    print(title_case("hello world"))  # Expected: Hello World
    print(title_case("python programming language"))  # Expected: Python Programming Language
    print()
    
    # Test is_palindrome
    print("Testing is_palindrome:")
    print(is_palindrome("racecar"))  # Expected: True
    print(is_palindrome("A man a plan a canal Panama"))  # Expected: True
    print(is_palindrome("hello"))  # Expected: False
    print()
    
    # Test count_letters
    print("Testing count_letters:")
    print(count_letters("hello world"))  # Expected: {'vowels': 3, 'consonants': 7}
    print(count_letters("PYTHON"))  # Expected: {'vowels': 1, 'consonants': 5}
    print()
    
    # Test format_phone_number
    print("Testing format_phone_number:")
    print(format_phone_number("1234567890"))  # Expected: (123) 456-7890
    print(format_phone_number("123-456-7890"))  # Expected: (123) 456-7890
    print(format_phone_number("123.456.7890"))  # Expected: (123) 456-7890
    print()
    
    # Test extract_words
    print("Testing extract_words:")
    print(extract_words("Hello, World! How are you?"))  # Expected: ['hello', 'world', 'how', 'are', 'you']
    print()
    
    # Test word_frequency
    print("Testing word_frequency:")
    print(word_frequency("hello world hello python"))  # Expected: {'hello': 2, 'world': 1, 'python': 1}
    print()
    
    # Test validate_password
    print("Testing validate_password:")
    print(validate_password("abc123"))  # Expected: {'valid': False, 'errors': [...]}
    print(validate_password("Password123!"))  # Expected: {'valid': True, 'errors': []}
    print()
    
    # Test caesar_cipher
    print("Testing caesar_cipher:")
    print(caesar_cipher("hello", 3))  # Expected: khoor
    print(caesar_cipher("khoor", -3))  # Expected: hello
    print() 