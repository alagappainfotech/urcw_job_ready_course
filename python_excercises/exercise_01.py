# exercise_01.py

def last_digit(n: int) -> int:
    """Returns the last digit of an integer."""
    try:
        n = int(n)  # Ensure n is an integer
    except ValueError:
        raise TypeError("Input must be an integer or convertible to an integer.")
    return abs(n) % 10

print(last_digit(100034)) # => 4
print(last_digit(-12345))  # => 5
print(last_digit("12345"))  # Raises TypeError
print(last_digit(0))  # => 0
print(last_digit(True))
print(last_digit(False))
print(last_digit("Hi There 123"))  # Raises TypeError