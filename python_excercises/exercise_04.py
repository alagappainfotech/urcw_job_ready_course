def max_of_numbers(*args: int) -> int:
    """Returns the maximum of the given numbers."""
    if not args:
        raise ValueError("At least one number must be provided")
    return max(args)

# Test cases for max_of_numbers function

print(max_of_numbers(1, 2, 3))  # => 3
print(max_of_numbers(-1, -2, -3))  # => -1
try:
    print(max_of_numbers())  # Raises ValueError
except ValueError as e:
    print(f"ValueError: {e}")
try:
    print(max_of_numbers("1", "2", "3"))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers(1, 2, "3"))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers(1, 2, 3.5))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers([1, 2, 3]))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers(None))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers({1: 'one', 2: 'two'}))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers((1, 2, 3)))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
try:
    print(max_of_numbers({1, 2, 3}))  # Raises TypeError
except TypeError as e:
    print(f"TypeError: {e}")
