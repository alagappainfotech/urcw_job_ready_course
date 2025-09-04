# Module 3: Code Organization, Functions & Error Handling

## Learning Objectives
By the end of this module, you will be able to:
- Structure Python programs with functions and modules effectively
- Understand scoping rules and namespaces in Python
- Master advanced function patterns and parameter handling
- Implement robust error management using exceptions
- Create reusable code components and libraries
- Apply best practices for code organization and documentation

## Core Concepts

### 3.1 Function Design Principles
Functions are the building blocks of well-organized Python programs. Key principles include:
- **Single Responsibility:** Each function should do one thing well
- **Pure Functions:** Functions that don't modify external state are easier to test and debug
- **Clear Interfaces:** Well-defined parameters and return values
- **Documentation:** Clear docstrings and type hints when appropriate

### 3.2 Python Scoping Rules
Understanding Python's scoping is crucial for avoiding bugs and writing maintainable code:
- **Local Scope:** Variables defined inside functions
- **Enclosing Scope:** Variables in outer functions (nonlocal)
- **Global Scope:** Variables at module level
- **Built-in Scope:** Python's built-in names

### 3.3 Error Handling Philosophy
Python follows the EAFP (Easier to Ask for Forgiveness than Permission) principle:
- **LBYL (Look Before You Leap):** Check conditions before operations
- **EAFP (Easier to Ask for Forgiveness than Permission):** Try operations and handle exceptions
- **Exception Hierarchy:** Understanding built-in exception types
- **Custom Exceptions:** Creating domain-specific error types

## Key Topics

### 3.4 Function Fundamentals
- **Function Definition:** def keyword, parameters, return statements
- **Parameter Types:** Positional, keyword, default, variable arguments
- **Return Values:** Single and multiple return values
- **Docstrings:** Function documentation standards
- **Type Hints:** Optional type annotations for better code clarity

### 3.5 Advanced Function Features
- **Variable Arguments:** *args and **kwargs
- **Lambda Functions:** Anonymous functions for simple operations
- **Higher-Order Functions:** Functions that take or return other functions
- **Closures:** Functions that capture variables from enclosing scope
- **Decorators:** Functions that modify other functions
- **Generator Functions:** Functions that yield values instead of returning

### 3.6 Module System
- **Module Creation:** Creating reusable code modules
- **Import System:** Different ways to import modules and functions
- **Package Structure:** Organizing code into packages
- **__init__.py:** Package initialization and public API definition
- **Module Search Path:** How Python finds modules
- **Private Names:** Convention for internal module components

### 3.7 Exception Handling
- **Exception Types:** Built-in exception hierarchy
- **Try-Except Blocks:** Catching and handling exceptions
- **Exception Chaining:** Preserving exception context
- **Custom Exceptions:** Creating domain-specific exceptions
- **Context Managers:** The with statement for resource management
- **Assertions:** Debugging and validation with assert

### 3.8 Code Organization Best Practices
- **Separation of Concerns:** Dividing code into logical components
- **DRY Principle:** Don't Repeat Yourself
- **SOLID Principles:** Object-oriented design principles
- **Code Documentation:** Comments, docstrings, and type hints
- **Testing:** Unit testing and test-driven development concepts

## Hands-on Learning Path

### Quick Checks (Immediate Reinforcement)
1. **Function Design:** Evaluate function signatures and responsibilities
2. **Scope Analysis:** Predict variable visibility in nested scopes
3. **Exception Handling:** Identify appropriate exception types
4. **Module Structure:** Design module and package organization

### Try This Exercises (Hands-on Application)
1. **Function Creation:** Build reusable utility functions
2. **Parameter Handling:** Practice with different parameter types
3. **Error Handling:** Implement robust exception handling
4. **Module Development:** Create and import custom modules

### Lab Problems (Critical Thinking)
1. **Text Processing Library:** Build a comprehensive text analysis module
2. **Data Validation System:** Create robust input validation functions
3. **Configuration Manager:** Design a flexible configuration system
4. **Logging Framework:** Implement a custom logging solution

## Assessment Criteria
- **Function Design:** Clear, single-purpose functions with good interfaces
- **Error Handling:** Comprehensive exception handling and validation
- **Code Organization:** Logical module structure and separation of concerns
- **Documentation:** Clear docstrings and comments
- **Reusability:** Functions and modules that can be easily reused
- **Testing:** Proper error handling and edge case coverage

## Resources
- [Python Functions Documentation](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Python Exception Handling](https://docs.python.org/3/tutorial/errors.html)
- [Python Modules Documentation](https://docs.python.org/3/tutorial/modules.html)

### Additional Topics
- `assert` usage and disabling with `-O`: demonstrate a small input check that is skipped when optimizations are enabled.
- `if __name__ == "__main__"`: explain purpose (prevent side effects on import), alternatives (CLI wrappers, `__main__.py`, entry points).
- Generator function quick check: convert a fixed-yield function into a parameterized generator.

### AI-comparison callout
For each lab, generate an AI solution and compare:
- Separation of concerns: functions vs script-level code
- Error handling scope and specificity
- Module import hygiene and namespacing

## Next Steps
After completing Module 3, you'll be ready to explore Python programs, filesystem interaction, and command-line interfaces in Module 4. The code organization skills you develop here will be essential for building larger applications.
