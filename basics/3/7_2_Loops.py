# Lesson: Python Loops
# Loops are used to execute a block of code repeatedly as long as a condition is met.

# 1. For Loop
# A `for` loop is used to iterate over a sequence (like a list, tuple, or string).

# Example: Iterating through a list
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)
# Output:
# apple
# banana
# cherry

# Example: Iterating through a range of numbers
for i in range(5):  # range(5) generates numbers 0 to 4
    print(i)
# Output:
# 0
# 1
# 2
# 3
# 4

# 2. While Loop
# A `while` loop runs as long as a condition is `True`.

# Example: Counting down from 5
count = 5
while count > 0:
    print(count)
    count -= 1
# Output:
# 5
# 4
# 3
# 2
# 1

# 3. Nested Loops
# Loops can be nested inside each other.

# Example: Printing a multiplication table
for i in range(1, 4):  # Outer loop
    for j in range(1, 4):  # Inner loop
        print(f"{i} x {j} = {i * j}")
# Output:
# 1 x 1 = 1
# 1 x 2 = 2
# 1 x 3 = 3
# 2 x 1 = 2
# ...

# 4. Break and Continue
# `break` exits the loop, and `continue` skips the current iteration.

# Example: Using `break`
for num in range(1, 10):
    if num == 5:
        break
    print(num)
# Output:
# 1
# 2
# 3
# 4

# Example: Using `continue`
for num in range(1, 6):
    if num == 3:
        continue
    print(num)
# Output:
# 1
# 2
# 4
# 5

# 5. Else with Loops
# The `else` block runs if the loop completes without a `break`.

# Example: Using `else` with a `for` loop
for num in range(1, 4):
    print(num)
else:
    print("Loop completed!")
# Output:
# 1
# 2
# 3
# Loop completed!

# Exercise:
# 1. Write a program to calculate the sum of all even numbers from 1 to 100.
# 2. Create a nested loop to print a pattern like:
#    *
#    **
#    ***
# 3. Write a program to find the first number divisible by 7 in a range of 1 to 50.