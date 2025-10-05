 MODULE-1
Quick Check 1: Pythonic Style Evaluation
==================================================
Evaluate these names for PEP 8 compliance:
  userName
  total_count
  MAX_SIZE
  getUserData
  x
  calculate_average_grade
  temp_var
  is_valid
  data_list
  process_data
Good names follow these rules:
- Use snake_case for variables and functions
- Use UPPER_CASE for constants
- Be descriptive but concise
- Avoid single letters (except in loops)
- Avoid redundant words (like 'data_list')

Quick Check 2: Type Identification
==================================================
Object -> Type -> Truthiness
------------------------------
      1234 -> int        -> True
      8.99 -> float      -> True
       9.0 -> float      -> True
      True -> bool       -> True
     False -> bool       -> False
   'hello' -> str        -> True
      None -> NoneType   -> False
 [1, 2, 3] -> list       -> True
 (1, 2, 3) -> tuple      -> True
 {1, 2, 3} -> set        -> True
 {'a': 1} -> dict       -> True
    (3+4j) -> complex    -> True

Quick Check 3: Operator Precedence
==================================================
Expression -> Result -> Explanation
--------------------------------------------------
2 + 3 * 4            -> 14       -> Multiplication before addition
(2 + 3) * 4          -> 20       -> Parentheses override precedence
2 ** 3 ** 2          -> 512      -> Right-associative exponentiation
(2 ** 3) ** 2        -> 64       -> Parentheses change associativity
10 / 3 * 3           -> 10.0     -> Left-to-right for same precedence
10 // 3 * 3          -> 9        -> Floor division vs regular division
not True and False   -> 0        -> not has higher precedence than and
not (True and False) -> 1        -> Parentheses change evaluation order

Try This 1: Getting Input and Type Conversion
==================================================

Testing input: '123'
Original type: str
As integer: 123
As float: 123.0
Contains only digits

Testing input: '45.67'
Original type: str
Cannot convert to integer
As float: 45.67
Contains only digits and one decimal point

Testing input: 'hello'
Original type: str
Cannot convert to integer
Cannot convert to float
Contains non-numeric characters

Testing input: ''
Original type: str
Cannot convert to integer
Cannot convert to float
Contains non-numeric characters

Testing input: '  42  '
Original type: str
As integer: 42
As float: 42.0
Contains only digits

Try This 2: Variable Manipulation
==================================================
Circle with radius 2.2:
  Area: 15.21
  Circumference: 13.82
Changing radius to 5.0...
New radius: 5.0
Area is still: 15.21 (not recalculated!)
Circumference is still: 13.82 (not recalculated!)
After recalculation:
  Area: 78.54
  Circumference: 31.42

Try This 3: String Operations
==================================================
Full name: John Doe
==============================
Escape sequences:
New line: Line 1\nLine 2
Tab: Column1\tColumn2
Quote: He said \"Hello\"
Backslash: Path: C:\\Users\\Name
Multi-line string:

    This is a multi-line string.
    It can span multiple lines.
    Perfect for documentation!
String formatting examples:
f-string: John is 25 years old
format(): John is 25 years old