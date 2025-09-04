# Module 5: Object-Oriented Programming & Advanced Features

## Learning Objectives
By the end of this module, you will be able to:
- Master Python's Object-Oriented Programming paradigm
- Understand advanced class features and inheritance patterns
- Implement special methods (dunder methods) effectively
- Master regular expressions and pattern matching
- Apply duck typing and advanced type checking
- Design and implement complex OOP architectures

## Core Concepts

### 5.1 Object-Oriented Programming Principles
Understanding the four pillars of OOP in Python:
- **Encapsulation:** Data hiding and access control
- **Inheritance:** Code reuse and hierarchical relationships
- **Polymorphism:** Multiple forms and method overriding
- **Abstraction:** Hiding complex implementation details

### 5.2 Python's OOP Implementation
Python's unique approach to object-oriented programming:
- **Everything is an Object:** Understanding Python's object model
- **Dynamic Typing:** Runtime type determination
- **Duck Typing:** "If it walks like a duck and quacks like a duck..."
- **Multiple Inheritance:** Complex inheritance hierarchies
- **Method Resolution Order (MRO):** Understanding inheritance resolution

### 5.3 Advanced Class Features
Mastering Python's advanced class capabilities:
- **Class Methods and Static Methods:** Different types of methods
- **Property Decorators:** Controlled attribute access
- **Descriptors:** Advanced attribute management
- **Metaclasses:** Classes that create classes
- **Abstract Base Classes:** Defining interfaces and contracts

### 5.4 Special Methods (Dunder Methods)
Understanding Python's magic methods:
- **Object Lifecycle:** `__init__`, `__new__`, `__del__`
- **String Representation:** `__str__`, `__repr__`
- **Comparison Operations:** `__eq__`, `__lt__`, `__hash__`
- **Arithmetic Operations:** `__add__`, `__mul__`, `__div__`
- **Container Operations:** `__len__`, `__getitem__`, `__setitem__`
- **Context Managers:** `__enter__`, `__exit__`

## Key Topics

### 5.5 Class Design and Implementation
- **Class Definition:** Creating classes with proper structure
- **Instance Variables:** Object-specific data storage
- **Class Variables:** Shared data across instances
- **Methods:** Instance, class, and static methods
- **Constructors and Destructors:** Object lifecycle management

### 5.6 Inheritance and Polymorphism
- **Single Inheritance:** Basic inheritance patterns
- **Multiple Inheritance:** Complex inheritance hierarchies
- **Method Overriding:** Customizing inherited behavior
- **Super() Function:** Accessing parent class methods
- **Method Resolution Order:** Understanding inheritance resolution

### 5.7 Advanced OOP Patterns
- **Composition vs. Inheritance:** Choosing the right relationship
- **Factory Pattern:** Creating objects through factories
- **Singleton Pattern:** Ensuring single instance
- **Observer Pattern:** Event-driven programming
- **Decorator Pattern:** Adding functionality to objects

### 5.8 Regular Expressions
- **Pattern Matching:** Using the `re` module
- **Special Characters:** Metacharacters and their meanings
- **Groups and Captures:** Extracting matched text
- **Substitution:** Replacing matched patterns
- **Performance Considerations:** Efficient regex usage

### 5.9 Type System and Duck Typing
- **Type Checking:** `isinstance()` and `issubclass()`
- **Duck Typing:** Interface-based programming
- **Protocols:** Informal interfaces
- **Type Hints:** Optional static typing
- **Generic Types:** Type parameters and constraints

## Hands-on Learning Path

### Quick Checks (Immediate Reinforcement)
1. **Class Design:** Evaluate class structure and relationships
2. **Inheritance Analysis:** Understand inheritance hierarchies
3. **Method Resolution:** Predict method call resolution
4. **Pattern Matching:** Design effective regular expressions

### Try This Exercises (Hands-on Application)
1. **Class Implementation:** Build complex class hierarchies
2. **Special Methods:** Implement custom object behavior
3. **Inheritance Patterns:** Create reusable base classes
4. **Regular Expressions:** Process text with complex patterns

### Lab Problems (Critical Thinking)
1. **Banking System:** Design a complete banking application
2. **Text Processing Engine:** Build a regex-based text processor
3. **Game Engine:** Create an object-oriented game framework
4. **Data Validation Framework:** Implement a flexible validation system

## Assessment Criteria
- **Class Design:** Well-structured, maintainable class hierarchies
- **Inheritance Usage:** Appropriate use of inheritance and composition
- **Special Methods:** Proper implementation of dunder methods
- **Code Reusability:** DRY principles and modular design
- **Error Handling:** Robust exception handling in OOP context
- **Performance:** Efficient object creation and method calls

## Resources
- [Python Classes Documentation](https://docs.python.org/3/tutorial/classes.html)
- [Python Special Methods](https://docs.python.org/3/reference/datamodel.html#special-method-names)
- [Regular Expressions Documentation](https://docs.python.org/3/library/re.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Python Design Patterns](https://python-patterns.guide/)

### Additional Labs to match requirements
- HTML classes lab: implement `Html`, `Body`, `P` with inheritance; `__str__` returns full document.
- `StringDict` subclass: enforce string-only keys/values by overriding `__setitem__` and validating in `__init__`.
- `TypedList` exercise: implement `__len__`, `__delitem__`, and `append` with type enforcement.

### AI-comparison callout
Compare OOP labs with AI-generated solutions for:
- Avoiding duplicated code via inheritance/composition
- Correct use of properties and special methods
- Robust validation and informative errors

## Next Steps
After completing Module 5, you'll be ready to explore packages, ecosystem, and project organization in Module 6. The OOP skills you develop here will be essential for building large-scale applications and frameworks.
