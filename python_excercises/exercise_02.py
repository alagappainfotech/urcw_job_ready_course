def first_digit(n: int) -> int:
    """Returns the first digit of an integer."""
    n = abs(n)
    while n >= 10:
        n //= 10
    return n
print(first_digit(100034))  # => 1
print(first_digit(-12345))  # => 1
print(first_digit(12345))   # => 1
print(first_digit(0))       # => 0
print(first_digit(True))    # => 1
print(first_digit(False))   # => 0

try:
    print(first_digit("12345"))  # Raises TypeError
    print(first_digit("Hi There 123"))  # Raises TypeError
    print(first_digit(3.14))  # Raises TypeError
    print(first_digit([1, 2, 3]))  # Raises TypeError
    print(first_digit(None))  # Raises TypeError
    print(first_digit({1: 'one', 2: 'two'}))  # Raises TypeError
    print(first_digit((1, 2, 3)))  # Raises TypeError
    print(first_digit({1, 2, 3}))  # Raises TypeError
    print(first_digit(bytearray(b'12345')))  # Raises TypeError
    print(first_digit(b'12345'))  # Raises TypeError
    print(first_digit(range(5)))  # Raises TypeError
    print(first_digit(3 + 4j))  # Raises TypeError
    print(first_digit(float('inf')))  # Raises TypeError
    print(first_digit(float('nan')))  # Raises TypeError
    print(first_digit(float(3.14)))  # Raises TypeError
    print(first_digit(complex(1, 2)))  # Raises TypeError
    print(first_digit(True))  # => 1
    print(first_digit(False))  # => 0
    print(first_digit(None))  # Raises TypeError
    print(first_digit("12345"))  # Raises TypeError

except TypeError as e:
    print(f"TypeError: {e}")
