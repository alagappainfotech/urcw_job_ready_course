class Car:
    def _init_(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def display_details(self):
        print(f"Make: {self.make}")
        print(f"Model: {self.model}")
        print(f"Year: {self.year}")
# Example usage
my_car = Car("Toyota", "Corolla", 2020)
my_car.display_details()

O/p
Make: Toyota
Model: Corolla
Year: 2020

class Rectangle:
    def _init_(self, length, width):
        self.length = length
        self.width = width

    def calculate_area(self):
        return self.length * self.width

    def calculate_perimeter(self):
        return 2 * (self.length + self.width)
# Example usage
my_rectangle = Rectangle(5, 3)
print(f"Area: {my_rectangle.calculate_area()}")
print(f"Perimeter: {my_rectangle.calculate_perimeter()}")

O/p
Area:15
Perimeter:16

class Shape:
    def _init_(self):
        pass

    def calculate_area(self):
        pass

    def calculate_perimeter(self):
        pass

class Square(Shape):
def _init_(self, side):
        self.side = side

    def calculate_area(self):
        return self.side ** 2

    def calculate_perimeter(self):
        return 4 * self.side

class Triangle(Shape):
    def _init_(self, side1, side2, side3):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3

    def calculate_perimeter(self):
return self.side1 + self.side2 + self.side3

    # For simplicity, let's assume it's a right-angled triangle
    def calculate_area(self, base, height):
        return 0.5 * base * height

# Example usage
my_square = Square(4)
print(f"Square Area: {my_square.calculate_area()}")
print(f"Square Perimeter: {my_square.calculate_perimeter()}")

my_triangle = Triangle(3, 4, 5)
print(f"Triangle Perimeter: {my_triangle.calculate_perimeter()}")
print(f"Triangle Area: {my_triangle.calculate_area(3, 4)}")
O/p

Square Area: 16
Square Perimeter: 16
Triangle Perimeter: 12
Triangle Area: 6.0