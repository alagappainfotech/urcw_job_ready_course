# Lesson: Python Classes
# Classes are blueprints for creating objects. They encapsulate data (attributes) and behavior (methods).

# 1. Defining a Class
class Person:
    """
    A simple class to represent a person.
    """
    # Constructor method to initialize attributes
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    # Method to display a greeting
    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Creating an object (instance) of the class
person1 = Person("Alice", 30)
person1.greet()  # Output: Hello, my name is Alice and I am 30 years old.

# 2. Class Attributes
class Circle:
    """
    A class to represent a circle.
    """
    pi = 3.14159  # Class attribute (shared by all instances)

    def __init__(self, radius):
        self.radius = radius  # Instance attribute

    def area(self):
        return Circle.pi * self.radius ** 2

circle1 = Circle(5)
print(circle1.area())  # Output: 78.53975

# 3. Inheritance
# A class can inherit attributes and methods from another class.
class Animal:
    def __init__(self, species):
        self.species = species

    def make_sound(self):
        print("Some generic sound")

class Dog(Animal):  # Dog inherits from Animal
    def __init__(self, name, breed):
        super().__init__("Dog")  # Call the parent class constructor
        self.name = name
        self.breed = breed

    def make_sound(self):
        print("Woof!")

dog1 = Dog("Buddy", "Golden Retriever")
dog1.make_sound()  # Output: Woof!

# 4. Encapsulation
# Use underscores to indicate private attributes or methods.
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        self.__balance += amount

    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount
        else:
            print("Insufficient funds")

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500

# 5. Polymorphism
# Different classes can have methods with the same name but different behavior.
class Cat(Animal):
    def make_sound(self):
        print("Meow!")

animals = [Dog("Buddy", "Golden Retriever"), Cat("Whiskers")]
for animal in animals:
    animal.make_sound()
# Output:
# Woof!
# Meow!

# Exercise:
# 1. Create a class `Car` with attributes `make`, `model`, and `year`, and a method to display details.
# 2. Create a class `Rectangle` with methods to calculate area and perimeter.
# 3. Create a parent class `Shape` and child classes `Square` and `Triangle` with appropriate methods.