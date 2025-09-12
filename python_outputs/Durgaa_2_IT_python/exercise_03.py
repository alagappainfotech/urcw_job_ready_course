def reverse_number(n: int):
    """Returns the integer with its digits reversed."""
    sign = -1 if n < 0 else 1
    reversed_str = str(abs(n))[::-1]
    return sign * int(reversed_str)

print(reverse_number(12345))
print(reverse_number(-12345))
print(reverse_number(100034))
print(reverse_number(0))
print(reverse_number(True))
print(reverse_number(False))

try:
    print(reverse_number("12345"))
    print(reverse_number("Hi There 123"))
    print(reverse_number(3.14))
    print(reverse_number([1, 2, 3]))
    print(reverse_number(None))
    print(reverse_number({1: 'one', 2: 'two'}))
    print(reverse_number((1, 2, 3)))
    print(reverse_number({1, 2, 3}))
    print(reverse_number(bytearray(b'12345')))
    print(reverse_number(b'12345'))
    print(reverse_number(range(5)))
    print(reverse_number(3 + 4j))
    print(reverse_number(float('inf')))
    print(reverse_number(float('nan')))
    print(reverse_number(float(3.14)))
    print(reverse_number(complex(1, 2)))
except TypeError as e:
    print(f"TypeError: {e}")


Output
54321
-54321
430001
0
1
0
TypeError: '<' not supported between instances of 'str' and 'int'