"""
Module 1: Python Absolute Basics & Foundational Data Types
Comprehensive Exercises and Lab Problems

This file contains hands-on exercises designed to reinforce the concepts
learned in Module 1. Complete these exercises to develop proficiency in
Python fundamentals.
"""

# =============================================================================
# EXERCISE SET 1: QUICK CHECKS - Immediate Reinforcement
# =============================================================================

def quick_check_1_pythonic_style():
    """
    Quick Check: Pythonic Style Evaluation
    
    Evaluate the following variable and function names against PEP 8 conventions.
    Identify which are good and which need improvement.
    """
    print("Quick Check 1: Pythonic Style Evaluation")
    print("=" * 50)
    
    # Example names to evaluate
    names_to_evaluate = [
        "userName",      # Should be user_name
        "total_count",   # Good
        "MAX_SIZE",      # Good for constants
        "getUserData",   # Should be get_user_data
        "x",             # Too short, not descriptive
        "calculate_average_grade",  # Good
        "temp_var",      # Could be more descriptive
        "is_valid",      # Good
        "data_list",     # Redundant, just 'data' is better
        "process_data"   # Good
    ]
    
    print("Evaluate these names for PEP 8 compliance:")
    for name in names_to_evaluate:
        print(f"  {name}")
    
    print("\nGood names follow these rules:")
    print("- Use snake_case for variables and functions")
    print("- Use UPPER_CASE for constants")
    print("- Be descriptive but concise")
    print("- Avoid single letters (except in loops)")
    print("- Avoid redundant words (like 'data_list')")


def quick_check_2_type_identification():
    """
    Quick Check: Type Identification
    
    Given various objects, determine their types and understand
    the implications of different data types.
    """
    print("\nQuick Check 2: Type Identification")
    print("=" * 50)
    
    # Test objects
    test_objects = [
        1234,           # int
        8.99,           # float
        9.0,            # float
        True,           # bool
        False,          # bool
        "hello",        # str
        None,           # NoneType
        [1, 2, 3],      # list
        (1, 2, 3),      # tuple
        {1, 2, 3},      # set
        {"a": 1},       # dict
        3 + 4j,         # complex
    ]
    
    print("Object -> Type -> Truthiness")
    print("-" * 30)
    
    for obj in test_objects:
        obj_type = type(obj).__name__
        truthiness = bool(obj)
        print(f"{obj!r:>10} -> {obj_type:<10} -> {truthiness}")


def quick_check_3_operator_precedence():
    """
    Quick Check: Operator Precedence
    
    Predict the results of complex expressions before running them.
    This helps develop intuition about Python's operator precedence.
    """
    print("\nQuick Check 3: Operator Precedence")
    print("=" * 50)
    
    expressions = [
        ("2 + 3 * 4", "Multiplication before addition"),
        ("(2 + 3) * 4", "Parentheses override precedence"),
        ("2 ** 3 ** 2", "Right-associative exponentiation"),
        ("(2 ** 3) ** 2", "Parentheses change associativity"),
        ("10 / 3 * 3", "Left-to-right for same precedence"),
        ("10 // 3 * 3", "Floor division vs regular division"),
        ("not True and False", "not has higher precedence than and"),
        ("not (True and False)", "Parentheses change evaluation order"),
    ]
    
    print("Expression -> Result -> Explanation")
    print("-" * 50)
    
    for expr, explanation in expressions:
        try:
            result = eval(expr)
            print(f"{expr:<20} -> {result:<8} -> {explanation}")
        except Exception as e:
            print(f"{expr:<20} -> ERROR: {e}")


# =============================================================================
# EXERCISE SET 2: TRY THIS - Hands-on Application
# =============================================================================

def try_this_1_input_and_conversion():
    """
    Try This: Getting Input and Type Conversion
    
    Experiment with input() function and type conversion.
    Handle different input scenarios gracefully.
    """
    print("\nTry This 1: Getting Input and Type Conversion")
    print("=" * 50)
    
    # Simulate different input scenarios
    test_inputs = ["123", "45.67", "hello", "", "  42  "]
    
    for user_input in test_inputs:
        print(f"\nTesting input: '{user_input}'")
        print(f"Original type: {type(user_input).__name__}")
        
        # Try converting to integer
        try:
            int_value = int(user_input)
            print(f"As integer: {int_value}")
        except ValueError:
            print("Cannot convert to integer")
        
        # Try converting to float
        try:
            float_value = float(user_input)
            print(f"As float: {float_value}")
        except ValueError:
            print("Cannot convert to float")
        
        # Check if it's a valid number (after stripping whitespace)
        stripped = user_input.strip()
        if stripped.isdigit():
            print("Contains only digits")
        elif stripped.replace('.', '').isdigit():
            print("Contains only digits and one decimal point")
        else:
            print("Contains non-numeric characters")


def try_this_2_variable_manipulation():
    """
    Try This: Variable Manipulation
    
    Practice binding values to variables and observe how changes
    affect different variables.
    """
    print("\nTry This 2: Variable Manipulation")
    print("=" * 50)
    
    # Example: Circle calculations
    pi = 3.14159
    radius = 2.2
    area = pi * radius ** 2
    circumference = 2 * pi * radius
    
    print(f"Circle with radius {radius}:")
    print(f"  Area: {area:.2f}")
    print(f"  Circumference: {circumference:.2f}")
    
    # Change radius and observe effects
    print(f"\nChanging radius to 5.0...")
    radius = 5.0
    print(f"New radius: {radius}")
    print(f"Area is still: {area:.2f} (not recalculated!)")
    print(f"Circumference is still: {circumference:.2f} (not recalculated!)")
    
    # Recalculate
    area = pi * radius ** 2
    circumference = 2 * pi * radius
    print(f"After recalculation:")
    print(f"  Area: {area:.2f}")
    print(f"  Circumference: {circumference:.2f}")


def try_this_3_string_operations():
    """
    Try This: String Operations
    
    Practice string manipulation, formatting, and escape sequences.
    """
    print("\nTry This 3: String Operations")
    print("=" * 50)
    
    # Basic string operations
    first_name = "John"
    last_name = "Doe"
    age = 25
    
    # String concatenation
    full_name = first_name + " " + last_name
    print(f"Full name: {full_name}")
    
    # String repetition
    separator = "=" * 30
    print(separator)
    
    # Escape sequences
    print("Escape sequences:")
    print("New line: Line 1\\nLine 2")
    print("Tab: Column1\\tColumn2")
    print("Quote: He said \\\"Hello\\\"")
    print("Backslash: Path: C:\\\\Users\\\\Name")
    
    # Triple-quoted strings
    multi_line = """
    This is a multi-line string.
    It can span multiple lines.
    Perfect for documentation!
    """
    print("Multi-line string:")
    print(multi_line)
    
    # String formatting
    print("String formatting examples:")
    print(f"f-string: {first_name} is {age} years old")
    print("format(): {} is {} years old".format(first_name, age))
    print("% formatting: %s is %d years old" % (first_name, age))


def try_this_4_value_swapping():
    """
    Try This: Value Swapping
    
    Practice different methods of swapping variable values.
    """
    print("\nTry This 4: Value Swapping")
    print("=" * 50)
    
    # Initial values
    a = 10
    b = 20
    print(f"Initial values: a = {a}, b = {b}")
    
    # Method 1: Using temporary variable
    temp = a
    a = b
    b = temp
    print(f"After temp swap: a = {a}, b = {b}")
    
    # Method 2: Python's tuple unpacking (most Pythonic)
    a, b = b, a
    print(f"After tuple unpacking: a = {a}, b = {b}")
    
    # Method 3: Using arithmetic (for numbers only)
    a = a + b
    b = a - b
    a = a - b
    print(f"After arithmetic swap: a = {a}, b = {b}")
    
    # Method 4: Using XOR (for integers only)
    a = a ^ b
    b = a ^ b
    a = a ^ b
    print(f"After XOR swap: a = {a}, b = {b}")


# =============================================================================
# EXERCISE SET 3: LAB PROBLEMS - Critical Thinking
# =============================================================================

def lab_1_temperature_converter():
    """
    Lab Problem 1: Temperature Converter
    
    Create a comprehensive temperature conversion system that handles
    Celsius, Fahrenheit, and Kelvin scales.
    """
    print("\nLab Problem 1: Temperature Converter")
    print("=" * 50)
    
    def celsius_to_fahrenheit(celsius):
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32
    
    def fahrenheit_to_celsius(fahrenheit):
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9
    
    def celsius_to_kelvin(celsius):
        """Convert Celsius to Kelvin."""
        return celsius + 273.15
    
    def kelvin_to_celsius(kelvin):
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15
    
    def fahrenheit_to_kelvin(fahrenheit):
        """Convert Fahrenheit to Kelvin."""
        return celsius_to_kelvin(fahrenheit_to_celsius(fahrenheit))
    
    def kelvin_to_fahrenheit(kelvin):
        """Convert Kelvin to Fahrenheit."""
        return celsius_to_fahrenheit(kelvin_to_celsius(kelvin))
    
    # Test the conversion functions
    test_temperatures = [0, 25, 100, -40, 37]  # Celsius
    
    print("Temperature Conversion Table:")
    print("Celsius | Fahrenheit | Kelvin")
    print("-" * 35)
    
    for celsius in test_temperatures:
        fahrenheit = celsius_to_fahrenheit(celsius)
        kelvin = celsius_to_kelvin(celsius)
        print(f"{celsius:7.1f} | {fahrenheit:10.1f} | {kelvin:6.1f}")
    
    # Test special cases
    print(f"\nSpecial cases:")
    print(f"Absolute zero: {kelvin_to_celsius(0):.2f}°C")
    print(f"Water freezing: {celsius_to_fahrenheit(0):.1f}°F")
    print(f"Water boiling: {celsius_to_fahrenheit(100):.1f}°F")
    print(f"Body temperature: {celsius_to_fahrenheit(37):.1f}°F")


def lab_2_enhanced_calculator():
    """
    Lab Problem 2: Enhanced Calculator
    
    Create a calculator that handles basic arithmetic operations
    with proper error handling and input validation.
    """
    print("\nLab Problem 2: Enhanced Calculator")
    print("=" * 50)
    
    def get_number(prompt):
        """Get a valid number from user input."""
        while True:
            try:
                value = float(input(prompt))
                return value
            except ValueError:
                print("Invalid input! Please enter a number.")
    
    def get_operation():
        """Get a valid operation from user."""
        valid_ops = ['+', '-', '*', '/', '**', '%']
        while True:
            op = input("Enter operation (+, -, *, /, **, %): ").strip()
            if op in valid_ops:
                return op
            print(f"Invalid operation! Choose from: {', '.join(valid_ops)}")
    
    def calculate(num1, num2, operation):
        """Perform calculation with error handling."""
        try:
            if operation == '+':
                return num1 + num2
            elif operation == '-':
                return num1 - num2
            elif operation == '*':
                return num1 * num2
            elif operation == '/':
                if num2 == 0:
                    raise ZeroDivisionError("Cannot divide by zero!")
                return num1 / num2
            elif operation == '**':
                return num1 ** num2
            elif operation == '%':
                if num2 == 0:
                    raise ZeroDivisionError("Cannot modulo by zero!")
                return num1 % num2
        except Exception as e:
            return f"Error: {e}"
    
    # Simulate calculator usage
    print("Enhanced Calculator")
    print("Enter 'quit' to exit")
    
    # Example calculations (simulated)
    test_cases = [
        (10, 3, '+'),
        (10, 3, '-'),
        (10, 3, '*'),
        (10, 3, '/'),
        (10, 3, '**'),
        (10, 3, '%'),
        (10, 0, '/'),  # Division by zero
    ]
    
    for num1, num2, op in test_cases:
        result = calculate(num1, num2, op)
        print(f"{num1} {op} {num2} = {result}")


def lab_3_grade_management_system():
    """
    Lab Problem 3: Grade Management System
    
    Create a system to calculate weighted averages and determine
    letter grades based on different criteria.
    """
    print("\nLab Problem 3: Grade Management System")
    print("=" * 50)
    
    def calculate_weighted_average(grades, weights):
        """Calculate weighted average of grades."""
        if len(grades) != len(weights):
            raise ValueError("Grades and weights must have the same length")
        
        if abs(sum(weights) - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError("Weights must sum to 1.0")
        
        return sum(grade * weight for grade, weight in zip(grades, weights))
    
    def get_letter_grade(percentage):
        """Convert percentage to letter grade."""
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_grade_point(letter_grade):
        """Convert letter grade to grade point."""
        grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
        return grade_points.get(letter_grade, 0.0)
    
    # Example student data
    students = [
        {
            'name': 'Alice',
            'grades': [85, 92, 78, 96],
            'weights': [0.25, 0.25, 0.25, 0.25]
        },
        {
            'name': 'Bob',
            'grades': [90, 88, 95, 87],
            'weights': [0.2, 0.3, 0.2, 0.3]
        },
        {
            'name': 'Charlie',
            'grades': [70, 75, 80, 72],
            'weights': [0.25, 0.25, 0.25, 0.25]
        }
    ]
    
    print("Grade Management System")
    print("=" * 30)
    
    for student in students:
        name = student['name']
        grades = student['grades']
        weights = student['weights']
        
        try:
            average = calculate_weighted_average(grades, weights)
            letter_grade = get_letter_grade(average)
            grade_point = get_grade_point(letter_grade)
            
            print(f"\n{name}:")
            print(f"  Grades: {grades}")
            print(f"  Weights: {weights}")
            print(f"  Average: {average:.2f}%")
            print(f"  Letter Grade: {letter_grade}")
            print(f"  Grade Point: {grade_point}")
            
        except ValueError as e:
            print(f"\n{name}: Error - {e}")


def lab_4_password_security_validator():
    """
    Lab Problem 4: Password Security Validator
    
    Create a comprehensive password validation system that checks
    multiple security criteria and provides detailed feedback.
    """
    print("\nLab Problem 4: Password Security Validator")
    print("=" * 50)
    
    def validate_password_strength(password):
        """
        Validate password strength and return detailed analysis.
        
        Returns:
            dict: Analysis results with criteria and overall score
        """
        criteria = {
            'length': len(password) >= 8,
            'has_uppercase': any(c.isupper() for c in password),
            'has_lowercase': any(c.islower() for c in password),
            'has_digit': any(c.isdigit() for c in password),
            'has_special': any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),
            'no_common_patterns': not any(pattern in password.lower() for pattern in 
                                        ['password', '123456', 'qwerty', 'abc123']),
            'no_repeated_chars': len(set(password)) >= len(password) * 0.6,
            'length_12_plus': len(password) >= 12
        }
        
        # Calculate strength score
        score = sum(criteria.values())
        max_score = len(criteria)
        
        # Determine strength level
        if score >= 7:
            strength = "Very Strong"
        elif score >= 5:
            strength = "Strong"
        elif score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"
        
        return {
            'criteria': criteria,
            'score': score,
            'max_score': max_score,
            'strength': strength,
            'percentage': (score / max_score) * 100
        }
    
    def display_password_analysis(password, analysis):
        """Display detailed password analysis."""
        print(f"\nPassword Analysis: '{password}'")
        print("-" * 40)
        
        criteria_descriptions = {
            'length': 'At least 8 characters',
            'has_uppercase': 'Contains uppercase letter',
            'has_lowercase': 'Contains lowercase letter',
            'has_digit': 'Contains digit',
            'has_special': 'Contains special character',
            'no_common_patterns': 'No common patterns',
            'no_repeated_chars': 'Sufficient character variety',
            'length_12_plus': 'At least 12 characters (bonus)'
        }
        
        for criterion, passed in analysis['criteria'].items():
            status = "✓" if passed else "✗"
            description = criteria_descriptions.get(criterion, criterion)
            print(f"  {status} {description}")
        
        print(f"\nStrength Score: {analysis['score']}/{analysis['max_score']} "
              f"({analysis['percentage']:.1f}%)")
        print(f"Overall Strength: {analysis['strength']}")
        
        # Provide recommendations
        if analysis['strength'] in ['Weak', 'Medium']:
            print("\nRecommendations:")
            for criterion, passed in analysis['criteria'].items():
                if not passed and criterion != 'length_12_plus':
                    description = criteria_descriptions.get(criterion, criterion)
                    print(f"  - {description}")
    
    # Test various passwords
    test_passwords = [
        "password",           # Weak
        "Password123",        # Medium
        "Password123!",       # Strong
        "MyStr0ng!P@ssw0rd",  # Very Strong
        "123456",            # Very Weak
        "qwerty",            # Very Weak
        "a",                 # Very Weak
        "Abc123!@#",         # Strong
    ]
    
    print("Password Security Validator")
    print("=" * 30)
    
    for password in test_passwords:
        analysis = validate_password_strength(password)
        display_password_analysis(password, analysis)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_exercises():
    """Run all exercises in sequence."""
    print("MODULE 1: PYTHON ABSOLUTE BASICS & FOUNDATIONAL DATA TYPES")
    print("COMPREHENSIVE EXERCISES")
    print("=" * 60)
    
    # Quick Checks
    quick_check_1_pythonic_style()
    quick_check_2_type_identification()
    quick_check_3_operator_precedence()
    
    # Try This Exercises
    try_this_1_input_and_conversion()
    try_this_2_variable_manipulation()
    try_this_3_string_operations()
    try_this_4_value_swapping()
    
    # Lab Problems
    lab_1_temperature_converter()
    lab_2_enhanced_calculator()
    lab_3_grade_management_system()
    lab_4_password_security_validator()
    
    print("\n" + "=" * 60)
    print("MODULE 1 EXERCISES COMPLETE!")
    print("Review your solutions and ensure they follow Pythonic best practices.")
    print("Next: Module 2 - Advanced Data Structures & Control Flow")
    print("=" * 60)


if __name__ == "__main__":
    run_all_exercises()
