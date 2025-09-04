"""
Module 3: Code Organization, Functions & Error Handling
Comprehensive Exercises and Lab Problems

This file contains hands-on exercises designed to reinforce the concepts
learned in Module 3. Complete these exercises to develop proficiency in
Python's function design, error handling, and code organization.
"""

# =============================================================================
# EXERCISE SET 1: QUICK CHECKS - Immediate Reinforcement
# =============================================================================

def quick_check_1_function_design():
    """
    Quick Check: Function Design
    
    Evaluate function signatures and responsibilities.
    Identify good and poor function design patterns.
    """
    print("Quick Check 1: Function Design")
    print("=" * 50)
    
    # Example functions to evaluate
    examples = [
        {
            "name": "Good Function",
            "code": "def calculate_rectangle_area(length: float, width: float) -> float:",
            "description": "Clear name, specific parameters, type hints, single responsibility",
            "rating": "Good"
        },
        {
            "name": "Poor Function",
            "code": "def do_stuff(x, y, z):",
            "description": "Vague name, unclear parameters, no type hints, unclear purpose",
            "rating": "Poor"
        },
        {
            "name": "Good Function",
            "code": "def validate_email(email: str) -> bool:",
            "description": "Clear purpose, specific parameter, type hints, boolean return",
            "rating": "Good"
        },
        {
            "name": "Poor Function",
            "code": "def process_data(data, config, options, flags, settings):",
            "description": "Too many parameters, unclear what each does, complex interface",
            "rating": "Poor"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  Code: {example['code']}")
        print(f"  Description: {example['description']}")
        print(f"  Rating: {example['rating']}")


def quick_check_2_scope_analysis():
    """
    Quick Check: Scope Analysis
    
    Predict variable visibility in nested scopes.
    Understand Python's LEGB (Local, Enclosing, Global, Built-in) rule.
    """
    print("\nQuick Check 2: Scope Analysis")
    print("=" * 50)
    
    # Global variable
    global_var = "I'm global"
    
    def outer_function():
        # Enclosing scope
        enclosing_var = "I'm in enclosing scope"
        
        def inner_function():
            # Local scope
            local_var = "I'm local"
            print(f"Local: {local_var}")
            print(f"Enclosing: {enclosing_var}")
            print(f"Global: {global_var}")
            print(f"Built-in: {len('test')}")  # len is built-in
        
        inner_function()
    
    outer_function()
    
    # Demonstrate nonlocal
    def counter_function():
        count = 0
        
        def increment():
            nonlocal count
            count += 1
            return count
        
        return increment
    
    counter = counter_function()
    print(f"\nCounter examples:")
    print(f"First call: {counter()}")
    print(f"Second call: {counter()}")
    print(f"Third call: {counter()}")


def quick_check_3_exception_handling():
    """
    Quick Check: Exception Handling
    
    Identify appropriate exception types for different scenarios.
    Understand exception hierarchy and when to use each type.
    """
    print("\nQuick Check 3: Exception Handling")
    print("=" * 50)
    
    exception_scenarios = [
        {
            "scenario": "File not found",
            "exception": "FileNotFoundError",
            "reason": "Specific exception for missing files"
        },
        {
            "scenario": "Division by zero",
            "exception": "ZeroDivisionError",
            "reason": "Specific exception for mathematical errors"
        },
        {
            "scenario": "Invalid user input",
            "exception": "ValueError",
            "reason": "General exception for invalid values"
        },
        {
            "scenario": "Network connection failed",
            "exception": "ConnectionError",
            "reason": "Specific exception for network issues"
        },
        {
            "scenario": "Index out of range",
            "exception": "IndexError",
            "reason": "Specific exception for sequence index errors"
        },
        {
            "scenario": "Key not found in dictionary",
            "exception": "KeyError",
            "reason": "Specific exception for missing dictionary keys"
        }
    ]
    
    for scenario in exception_scenarios:
        print(f"Scenario: {scenario['scenario']}")
        print(f"  Exception: {scenario['exception']}")
        print(f"  Reason: {scenario['reason']}\n")


def quick_check_4_module_structure():
    """
    Quick Check: Module Structure
    
    Design module and package organization.
    Understand import patterns and best practices.
    """
    print("\nQuick Check 4: Module Structure")
    print("=" * 50)
    
    print("Recommended module structure:")
    print("""
    my_package/
    ├── __init__.py          # Package initialization
    ├── core/                # Core functionality
    │   ├── __init__.py
    │   ├── models.py        # Data models
    │   └── utils.py         # Utility functions
    ├── api/                 # API interfaces
    │   ├── __init__.py
    │   └── endpoints.py
    ├── tests/               # Test modules
    │   ├── __init__.py
    │   ├── test_models.py
    │   └── test_utils.py
    └── docs/                # Documentation
        └── README.md
    """)
    
    print("Import best practices:")
    print("1. Use absolute imports: from my_package.core import utils")
    print("2. Import specific functions: from my_package.core.utils import helper_func")
    print("3. Use __all__ in __init__.py to control public API")
    print("4. Avoid circular imports")
    print("5. Use relative imports within packages: from .utils import helper")


# =============================================================================
# EXERCISE SET 2: TRY THIS - Hands-on Application
# =============================================================================

def try_this_1_function_creation():
    """
    Try This: Function Creation
    
    Build reusable utility functions with proper design principles.
    """
    print("\nTry This 1: Function Creation")
    print("=" * 50)
    
    def format_currency(amount: float, currency: str = "USD", precision: int = 2) -> str:
        """
        Format a number as currency.
        
        Args:
            amount: The amount to format
            currency: Currency code (default: USD)
            precision: Number of decimal places (default: 2)
        
        Returns:
            Formatted currency string
        """
        currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        symbol = currency_symbols.get(currency, currency)
        return f"{symbol}{amount:,.{precision}f}"
    
    def calculate_percentage(part: float, whole: float) -> float:
        """
        Calculate percentage of part relative to whole.
        
        Args:
            part: The part value
            whole: The whole value
        
        Returns:
            Percentage as a float
        
        Raises:
            ValueError: If whole is zero
        """
        if whole == 0:
            raise ValueError("Cannot calculate percentage with zero whole value")
        return (part / whole) * 100
    
    def is_valid_email(email: str) -> bool:
        """
        Basic email validation.
        
        Args:
            email: Email address to validate
        
        Returns:
            True if email appears valid, False otherwise
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    # Test the functions
    print("Currency formatting:")
    print(f"  ${format_currency(1234.56)}")
    print(f"  {format_currency(1234.56, 'EUR')}")
    print(f"  {format_currency(1234.56, 'JPY', 0)}")
    
    print("\nPercentage calculation:")
    print(f"  25 out of 100 = {calculate_percentage(25, 100):.1f}%")
    print(f"  3 out of 10 = {calculate_percentage(3, 10):.1f}%")
    
    print("\nEmail validation:")
    test_emails = ["user@example.com", "invalid.email", "test@domain.co.uk", "not-an-email"]
    for email in test_emails:
        print(f"  {email}: {is_valid_email(email)}")


def try_this_2_parameter_handling():
    """
    Try This: Parameter Handling
    
    Practice with different parameter types and patterns.
    """
    print("\nTry This 2: Parameter Handling")
    print("=" * 50)
    
    def flexible_function(required_arg, *args, keyword_arg="default", **kwargs):
        """
        Demonstrate different parameter types.
        
        Args:
            required_arg: A required positional argument
            *args: Variable number of positional arguments
            keyword_arg: A keyword argument with default value
            **kwargs: Variable number of keyword arguments
        """
        print(f"Required argument: {required_arg}")
        print(f"Variable args: {args}")
        print(f"Keyword argument: {keyword_arg}")
        print(f"Variable kwargs: {kwargs}")
    
    # Test different calling patterns
    print("Call 1 - Basic usage:")
    flexible_function("hello", 1, 2, 3, keyword_arg="custom")
    
    print("\nCall 2 - With kwargs:")
    flexible_function("world", 4, 5, keyword_arg="test", extra1="value1", extra2="value2")
    
    print("\nCall 3 - All defaults:")
    flexible_function("defaults")
    
    # Function with type hints and validation
    def process_numbers(*numbers: int, operation: str = "sum") -> float:
        """
        Process a variable number of integers.
        
        Args:
            *numbers: Variable number of integers
            operation: Operation to perform ('sum', 'avg', 'max', 'min')
        
        Returns:
            Result of the operation
        """
        if not numbers:
            raise ValueError("At least one number is required")
        
        if operation == "sum":
            return sum(numbers)
        elif operation == "avg":
            return sum(numbers) / len(numbers)
        elif operation == "max":
            return max(numbers)
        elif operation == "min":
            return min(numbers)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    print("\nNumber processing:")
    print(f"Sum: {process_numbers(1, 2, 3, 4, 5)}")
    print(f"Average: {process_numbers(1, 2, 3, 4, 5, operation='avg')}")
    print(f"Max: {process_numbers(1, 2, 3, 4, 5, operation='max')}")
    print(f"Min: {process_numbers(1, 2, 3, 4, 5, operation='min')}")


def try_this_3_error_handling():
    """
    Try This: Error Handling
    
    Implement robust exception handling patterns.
    """
    print("\nTry This 3: Error Handling")
    print("=" * 50)
    
    def safe_file_operation(filename: str, operation: str = "read") -> str:
        """
        Safely perform file operations with comprehensive error handling.
        
        Args:
            filename: Name of the file
            operation: Operation to perform ('read', 'write')
        
        Returns:
            File content (for read) or success message (for write)
        """
        try:
            if operation == "read":
                with open(filename, 'r') as file:
                    return file.read()
            elif operation == "write":
                with open(filename, 'w') as file:
                    file.write("Sample content")
                return f"Successfully wrote to {filename}"
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        except FileNotFoundError:
            return f"Error: File '{filename}' not found"
        except PermissionError:
            return f"Error: Permission denied for '{filename}'"
        except IsADirectoryError:
            return f"Error: '{filename}' is a directory, not a file"
        except Exception as e:
            return f"Unexpected error: {type(e).__name__}: {e}"
    
    # Test error handling
    print("File operations:")
    print(f"  {safe_file_operation('nonexistent.txt')}")
    print(f"  {safe_file_operation('test.txt', 'write')}")
    print(f"  {safe_file_operation('test.txt', 'read')}")
    print(f"  {safe_file_operation('test.txt', 'invalid')}")
    
    # Custom exception handling
    class ValidationError(Exception):
        """Custom exception for validation errors."""
        pass
    
    def validate_user_data(name: str, age: int, email: str) -> bool:
        """
        Validate user data with custom exceptions.
        
        Args:
            name: User's name
            age: User's age
            email: User's email
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If validation fails
        """
        if not name or len(name.strip()) < 2:
            raise ValidationError("Name must be at least 2 characters long")
        
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ValidationError("Age must be an integer between 0 and 150")
        
        if not email or '@' not in email:
            raise ValidationError("Email must contain '@' symbol")
        
        return True
    
    # Test validation
    test_cases = [
        ("Alice", 25, "alice@example.com"),
        ("", 25, "alice@example.com"),
        ("Bob", -5, "bob@example.com"),
        ("Charlie", 30, "invalid-email")
    ]
    
    print("\nUser validation:")
    for name, age, email in test_cases:
        try:
            validate_user_data(name, age, email)
            print(f"  ✓ Valid: {name}, {age}, {email}")
        except ValidationError as e:
            print(f"  ✗ Invalid: {name}, {age}, {email} - {e}")


def try_this_4_module_development():
    """
    Try This: Module Development
    
    Create and import custom modules.
    """
    print("\nTry This 4: Module Development")
    print("=" * 50)
    
    # Simulate a module (in real code, this would be in a separate file)
    class StringUtils:
        """Utility functions for string operations."""
        
        @staticmethod
        def reverse_string(text: str) -> str:
            """Reverse a string."""
            return text[::-1]
        
        @staticmethod
        def count_words(text: str) -> int:
            """Count words in a string."""
            return len(text.split())
        
        @staticmethod
        def is_palindrome(text: str) -> bool:
            """Check if a string is a palindrome."""
            cleaned = text.lower().replace(" ", "")
            return cleaned == cleaned[::-1]
    
    class MathUtils:
        """Utility functions for mathematical operations."""
        
        @staticmethod
        def is_prime(n: int) -> bool:
            """Check if a number is prime."""
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        @staticmethod
        def fibonacci(n: int) -> list:
            """Generate first n Fibonacci numbers."""
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            elif n == 2:
                return [0, 1]
            
            fib = [0, 1]
            for i in range(2, n):
                fib.append(fib[i-1] + fib[i-2])
            return fib
    
    # Using the modules
    print("String utilities:")
    text = "Hello World"
    print(f"  Original: {text}")
    print(f"  Reversed: {StringUtils.reverse_string(text)}")
    print(f"  Word count: {StringUtils.count_words(text)}")
    print(f"  Is palindrome: {StringUtils.is_palindrome('racecar')}")
    
    print("\nMath utilities:")
    print(f"  Is 17 prime: {MathUtils.is_prime(17)}")
    print(f"  Is 15 prime: {MathUtils.is_prime(15)}")
    print(f"  First 10 Fibonacci numbers: {MathUtils.fibonacci(10)}")


def try_this_5_assert_and_generators():
    """
    Try This: assert usage and generator quick check
    """
    print("\nTry This 5: assert and generators")
    print("=" * 50)

    def positive_only(n: int) -> int:
        assert n >= 0, "n must be non-negative"
        return n

    # Demonstrate generator conversion
    def count_up_to(n: int):
        for i in range(n + 1):
            yield i

    print("count_up_to(5):", list(count_up_to(5)))
    try:
        positive_only(-1)
    except AssertionError as e:
        print("Assertion triggered:", e)


# =============================================================================
# EXERCISE SET 3: LAB PROBLEMS - Critical Thinking
# =============================================================================

def lab_1_text_processing_library():
    """
    Lab Problem 1: Text Processing Library
    
    Build a comprehensive text analysis module with multiple functions
    for different text processing tasks.
    """
    print("\nLab Problem 1: Text Processing Library")
    print("=" * 50)
    
    class TextProcessor:
        """Comprehensive text processing library."""
        
        @staticmethod
        def clean_text(text: str, remove_punctuation: bool = True, 
                      to_lowercase: bool = True) -> str:
            """
            Clean and normalize text.
            
            Args:
                text: Input text to clean
                remove_punctuation: Whether to remove punctuation
                to_lowercase: Whether to convert to lowercase
            
            Returns:
                Cleaned text
            """
            import string
            
            if to_lowercase:
                text = text.lower()
            
            if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        
        @staticmethod
        def extract_words(text: str) -> list:
            """Extract individual words from text."""
            cleaned = TextProcessor.clean_text(text)
            return [word for word in cleaned.split() if word]
        
        @staticmethod
        def word_frequency(text: str) -> dict:
            """Calculate word frequency in text."""
            words = TextProcessor.extract_words(text)
            frequency = {}
            for word in words:
                frequency[word] = frequency.get(word, 0) + 1
            return frequency
        
        @staticmethod
        def most_common_words(text: str, n: int = 5) -> list:
            """Find most common words in text."""
            frequency = TextProcessor.word_frequency(text)
            return sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:n]
        
        @staticmethod
        def text_statistics(text: str) -> dict:
            """Generate comprehensive text statistics."""
            words = TextProcessor.extract_words(text)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            return {
                'character_count': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': len(set(words)),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            }
        
        @staticmethod
        def find_keywords(text: str, keywords: list) -> dict:
            """Find occurrences of specific keywords in text."""
            cleaned_text = TextProcessor.clean_text(text)
            found_keywords = {}
            
            for keyword in keywords:
                count = cleaned_text.count(keyword.lower())
                if count > 0:
                    found_keywords[keyword] = count
            
            return found_keywords
    
    # Test the text processing library
    sample_text = """
    Python is a powerful programming language. Python is widely used for 
    web development, data science, and automation. Python has a simple 
    syntax that makes it easy to learn. Many companies use Python for 
    their backend systems. Python's extensive library ecosystem makes it 
    very powerful for various applications.
    """
    
    processor = TextProcessor()
    
    print("Text Processing Library Demo")
    print("-" * 40)
    
    # Clean text
    cleaned = processor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned[:100]}...")
    
    # Extract words
    words = processor.extract_words(sample_text)
    print(f"Extracted words: {words[:10]}...")
    
    # Word frequency
    frequency = processor.word_frequency(sample_text)
    print(f"Word frequency: {dict(list(frequency.items())[:5])}")
    
    # Most common words
    common = processor.most_common_words(sample_text, 3)
    print(f"Most common words: {common}")
    
    # Text statistics
    stats = processor.text_statistics(sample_text)
    print(f"Text statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Find keywords
    keywords = processor.find_keywords(sample_text, ["python", "programming", "development"])
    print(f"Found keywords: {keywords}")


def lab_2_data_validation_system():
    """
    Lab Problem 2: Data Validation System
    
    Create a robust input validation system with custom exceptions
    and comprehensive validation rules.
    """
    print("\nLab Problem 2: Data Validation System")
    print("=" * 50)
    
    class ValidationError(Exception):
        """Base exception for validation errors."""
        pass
    
    class DataValidator:
        """Comprehensive data validation system."""
        
        @staticmethod
        def validate_email(email: str) -> bool:
            """Validate email address format."""
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, email):
                raise ValidationError(f"Invalid email format: {email}")
            return True
        
        @staticmethod
        def validate_phone(phone: str) -> bool:
            """Validate phone number format."""
            import re
            # Remove all non-digit characters
            digits = re.sub(r'\D', '', phone)
            
            if len(digits) < 10 or len(digits) > 15:
                raise ValidationError(f"Invalid phone number length: {phone}")
            
            return True
        
        @staticmethod
        def validate_age(age: int) -> bool:
            """Validate age range."""
            if not isinstance(age, int):
                raise ValidationError(f"Age must be an integer: {age}")
            if age < 0 or age > 150:
                raise ValidationError(f"Age must be between 0 and 150: {age}")
            return True
        
        @staticmethod
        def validate_password(password: str) -> bool:
            """Validate password strength."""
            if len(password) < 8:
                raise ValidationError("Password must be at least 8 characters long")
            
            if not any(c.isupper() for c in password):
                raise ValidationError("Password must contain at least one uppercase letter")
            
            if not any(c.islower() for c in password):
                raise ValidationError("Password must contain at least one lowercase letter")
            
            if not any(c.isdigit() for c in password):
                raise ValidationError("Password must contain at least one digit")
            
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                raise ValidationError("Password must contain at least one special character")
            
            return True
        
        @staticmethod
        def validate_user_data(user_data: dict) -> dict:
            """
            Validate complete user data.
            
            Args:
                user_data: Dictionary containing user information
            
            Returns:
                Dictionary with validation results
            """
            results = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Required fields
            required_fields = ['name', 'email', 'age']
            for field in required_fields:
                if field not in user_data:
                    results['errors'].append(f"Missing required field: {field}")
                    results['valid'] = False
            
            if not results['valid']:
                return results
            
            # Validate individual fields
            try:
                DataValidator.validate_email(user_data['email'])
            except ValidationError as e:
                results['errors'].append(str(e))
                results['valid'] = False
            
            try:
                DataValidator.validate_age(user_data['age'])
            except ValidationError as e:
                results['errors'].append(str(e))
                results['valid'] = False
            
            # Optional field validation
            if 'phone' in user_data:
                try:
                    DataValidator.validate_phone(user_data['phone'])
                except ValidationError as e:
                    results['warnings'].append(str(e))
            
            if 'password' in user_data:
                try:
                    DataValidator.validate_password(user_data['password'])
                except ValidationError as e:
                    results['errors'].append(str(e))
                    results['valid'] = False
            
            return results
    
    # Test the validation system
    validator = DataValidator()
    
    print("Data Validation System Demo")
    print("-" * 40)
    
    # Test individual validations
    test_emails = ["user@example.com", "invalid-email", "test@domain.co.uk"]
    for email in test_emails:
        try:
            validator.validate_email(email)
            print(f"✓ Valid email: {email}")
        except ValidationError as e:
            print(f"✗ Invalid email: {email} - {e}")
    
    # Test complete user data validation
    test_users = [
        {
            'name': 'Alice Johnson',
            'email': 'alice@example.com',
            'age': 25,
            'phone': '+1-555-123-4567',
            'password': 'SecurePass123!'
        },
        {
            'name': 'Bob Smith',
            'email': 'invalid-email',
            'age': -5,
            'password': 'weak'
        },
        {
            'name': 'Charlie Brown',
            'email': 'charlie@example.com',
            'age': 30
            # Missing phone and password
        }
    ]
    
    print(f"\nUser data validation:")
    for i, user in enumerate(test_users, 1):
        print(f"\nUser {i}: {user.get('name', 'Unknown')}")
        results = validator.validate_user_data(user)
        
        if results['valid']:
            print("  ✓ Valid user data")
        else:
            print("  ✗ Invalid user data")
        
        if results['errors']:
            print("  Errors:")
            for error in results['errors']:
                print(f"    - {error}")
        
        if results['warnings']:
            print("  Warnings:")
            for warning in results['warnings']:
                print(f"    - {warning}")


def lab_3_configuration_manager():
    """
    Lab Problem 3: Configuration Manager
    
    Design a flexible configuration system that can load settings
    from different sources and validate configuration data.
    """
    print("\nLab Problem 3: Configuration Manager")
    print("=" * 50)
    
    import json
    from typing import Dict, Any, Optional
    
    class ConfigurationError(Exception):
        """Exception for configuration-related errors."""
        pass
    
    class ConfigurationManager:
        """Flexible configuration management system."""
        
        def __init__(self):
            self._config = {}
            self._defaults = {}
            self._validators = {}
        
        def set_default(self, key: str, value: Any) -> None:
            """Set a default configuration value."""
            self._defaults[key] = value
        
        def set_validator(self, key: str, validator_func: callable) -> None:
            """Set a validation function for a configuration key."""
            self._validators[key] = validator_func
        
        def set_config(self, key: str, value: Any) -> None:
            """Set a configuration value with validation."""
            # Apply default if not set
            if key not in self._config and key in self._defaults:
                self._config[key] = self._defaults[key]
            
            # Validate if validator exists
            if key in self._validators:
                try:
                    self._validators[key](value)
                except Exception as e:
                    raise ConfigurationError(f"Validation failed for '{key}': {e}")
            
            self._config[key] = value
        
        def get_config(self, key: str, default: Any = None) -> Any:
            """Get a configuration value."""
            return self._config.get(key, self._defaults.get(key, default))
        
        def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
            """Load configuration from a dictionary."""
            for key, value in config_dict.items():
                self.set_config(key, value)
        
        def load_from_json(self, json_string: str) -> None:
            """Load configuration from JSON string."""
            try:
                config_dict = json.loads(json_string)
                self.load_from_dict(config_dict)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON: {e}")
        
        def to_dict(self) -> Dict[str, Any]:
            """Export configuration as dictionary."""
            return self._config.copy()
        
        def validate_all(self) -> Dict[str, list]:
            """Validate all configuration values."""
            errors = {}
            
            for key, value in self._config.items():
                if key in self._validators:
                    try:
                        self._validators[key](value)
                    except Exception as e:
                        errors[key] = [str(e)]
            
            return errors
    
    # Create configuration manager
    config = ConfigurationManager()
    
    # Set up defaults
    config.set_default('database_host', 'localhost')
    config.set_default('database_port', 5432)
    config.set_default('debug_mode', False)
    config.set_default('max_connections', 100)
    
    # Set up validators
    def validate_port(port):
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("Port must be an integer between 1 and 65535")
    
    def validate_max_connections(max_conn):
        if not isinstance(max_conn, int) or max_conn < 1:
            raise ValueError("Max connections must be a positive integer")
    
    def validate_debug_mode(debug):
        if not isinstance(debug, bool):
            raise ValueError("Debug mode must be a boolean")
    
    config.set_validator('database_port', validate_port)
    config.set_validator('max_connections', validate_max_connections)
    config.set_validator('debug_mode', validate_debug_mode)
    
    # Load configuration
    sample_config = {
        'database_host': 'prod-db.example.com',
        'database_port': 5432,
        'database_name': 'myapp',
        'debug_mode': True,
        'max_connections': 200,
        'api_timeout': 30
    }
    
    print("Configuration Manager Demo")
    print("-" * 40)
    
    # Load configuration
    config.load_from_dict(sample_config)
    
    # Display configuration
    print("Current configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    errors = config.validate_all()
    if errors:
        print(f"\nValidation errors:")
        for key, error_list in errors.items():
            print(f"  {key}: {', '.join(error_list)}")
    else:
        print(f"\n✓ All configuration values are valid")
    
    # Test JSON loading
    json_config = '''
    {
        "database_host": "test-db.example.com",
        "database_port": 3306,
        "debug_mode": false,
        "max_connections": 50
    }
    '''
    
    print(f"\nLoading from JSON:")
    try:
        config.load_from_json(json_config)
        print("✓ JSON configuration loaded successfully")
        print(f"  Database host: {config.get_config('database_host')}")
        print(f"  Database port: {config.get_config('database_port')}")
    except ConfigurationError as e:
        print(f"✗ Error loading JSON: {e}")


def lab_4_logging_framework():
    """
    Lab Problem 4: Logging Framework
    
    Implement a custom logging solution with different log levels,
    formatters, and output destinations.
    """
    print("\nLab Problem 4: Logging Framework")
    print("=" * 50)
    
    import datetime
    from enum import Enum
    from typing import List, Optional
    
    class LogLevel(Enum):
        """Log levels in order of severity."""
        DEBUG = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
        CRITICAL = 4
    
    class LogFormatter:
        """Base class for log formatters."""
        
        def format(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> str:
            """Format a log message."""
            raise NotImplementedError
    
    class SimpleFormatter(LogFormatter):
        """Simple log formatter."""
        
        def format(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> str:
            return f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level.name}: {message}"
    
    class DetailedFormatter(LogFormatter):
        """Detailed log formatter."""
        
        def format(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> str:
            return f"[{timestamp.isoformat()}] [{level.name:8}] {message}"
    
    class LogHandler:
        """Base class for log handlers."""
        
        def __init__(self, formatter: LogFormatter):
            self.formatter = formatter
        
        def handle(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> None:
            """Handle a log message."""
            raise NotImplementedError
    
    class ConsoleHandler(LogHandler):
        """Console log handler."""
        
        def handle(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> None:
            formatted = self.formatter.format(level, message, timestamp)
            print(formatted)
    
    class FileHandler(LogHandler):
        """File log handler."""
        
        def __init__(self, filename: str, formatter: LogFormatter):
            super().__init__(formatter)
            self.filename = filename
        
        def handle(self, level: LogLevel, message: str, timestamp: datetime.datetime) -> None:
            formatted = self.formatter.format(level, message, timestamp)
            with open(self.filename, 'a') as f:
                f.write(formatted + '\n')
    
    class Logger:
        """Custom logging framework."""
        
        def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
            self.name = name
            self.level = level
            self.handlers: List[LogHandler] = []
        
        def add_handler(self, handler: LogHandler) -> None:
            """Add a log handler."""
            self.handlers.append(handler)
        
        def set_level(self, level: LogLevel) -> None:
            """Set the minimum log level."""
            self.level = level
        
        def _log(self, level: LogLevel, message: str) -> None:
            """Internal logging method."""
            if level.value >= self.level.value:
                timestamp = datetime.datetime.now()
                for handler in self.handlers:
                    handler.handle(level, message, timestamp)
        
        def debug(self, message: str) -> None:
            """Log a debug message."""
            self._log(LogLevel.DEBUG, message)
        
        def info(self, message: str) -> None:
            """Log an info message."""
            self._log(LogLevel.INFO, message)
        
        def warning(self, message: str) -> None:
            """Log a warning message."""
            self._log(LogLevel.WARNING, message)
        
        def error(self, message: str) -> None:
            """Log an error message."""
            self._log(LogLevel.ERROR, message)
        
        def critical(self, message: str) -> None:
            """Log a critical message."""
            self._log(LogLevel.CRITICAL, message)
    
    # Test the logging framework
    print("Logging Framework Demo")
    print("-" * 40)
    
    # Create logger
    logger = Logger("MyApp", LogLevel.DEBUG)
    
    # Add handlers
    console_handler = ConsoleHandler(SimpleFormatter())
    logger.add_handler(console_handler)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("Application started successfully")
    logger.warning("This is a warning message")
    logger.error("An error occurred")
    logger.critical("Critical system failure")
    
    # Test with different log level
    print(f"\nTesting with WARNING level:")
    logger.set_level(LogLevel.WARNING)
    logger.debug("This debug message won't be shown")
    logger.info("This info message won't be shown")
    logger.warning("This warning will be shown")
    logger.error("This error will be shown")
    
    # Test file handler
    print(f"\nTesting file handler:")
    file_handler = FileHandler("test.log", DetailedFormatter())
    logger.add_handler(file_handler)
    logger.set_level(LogLevel.INFO)
    logger.info("This message will be written to file")
    logger.warning("This warning will also be written to file")
    
    print("Check 'test.log' file for detailed log entries")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_exercises():
    """Run all exercises in sequence."""
    print("MODULE 3: CODE ORGANIZATION, FUNCTIONS & ERROR HANDLING")
    print("COMPREHENSIVE EXERCISES")
    print("=" * 60)
    
    # Quick Checks
    quick_check_1_function_design()
    quick_check_2_scope_analysis()
    quick_check_3_exception_handling()
    quick_check_4_module_structure()
    
    # Try This Exercises
    try_this_1_function_creation()
    try_this_2_parameter_handling()
    try_this_3_error_handling()
    try_this_4_module_development()
    try_this_5_assert_and_generators()
    
    # Lab Problems
    lab_1_text_processing_library()
    lab_2_data_validation_system()
    lab_3_configuration_manager()
    lab_4_logging_framework()
    
    print("\n" + "=" * 60)
    print("MODULE 3 EXERCISES COMPLETE!")
    print("Review your solutions and ensure they follow Pythonic best practices.")
    print("Next: Module 4 - Python Programs & Filesystem Interaction")
    print("=" * 60)


if __name__ == "__main__":
    run_all_exercises()
