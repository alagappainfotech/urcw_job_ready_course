def reverse_number(n: int) -> int:
    """Returns the integer with its digits reversed."""
    sign = -1 if n < 0 else 1
    reversed_str = str(abs(n))[::-1]
    return sign * int(reversed_str)

# Test cases for reverse_number function
print(reverse_number(12345))  # => 54321
print(reverse_number(-12345))  # => -54321
print(reverse_number(100034))  # => 430001
print(reverse_number(0))       # => 0
print(reverse_number(True))    # => 1
print(reverse_number(False))   # => 0

try:
    print(reverse_number("12345"))  # Raises TypeError
    print(reverse_number("Hi There 123"))  # Raises TypeError
    print(reverse_number(3.14))  # Raises TypeError
    print(reverse_number([1, 2, 3]))  # Raises TypeError
    print(reverse_number(None))  # Raises TypeError
    print(reverse_number({1: 'one', 2: 'two'}))  # Raises TypeError
    print(reverse_number((1, 2, 3)))  # Raises TypeError
    print(reverse_number({1, 2, 3}))  # Raises TypeError
    print(reverse_number(bytearray(b'12345')))  # Raises TypeError
    print(reverse_number(b'12345'))  # Raises TypeError
    print(reverse_number(range(5)))  # Raises TypeError
    print(reverse_number(3 + 4j))  # Raises TypeError
    print(reverse_number(float('inf')))  # Raises TypeError
    print(reverse_number(float('nan')))  # Raises TypeError
    print(reverse_number(float(3.14)))  # Raises TypeError
    print(reverse_number(complex(1, 2)))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")