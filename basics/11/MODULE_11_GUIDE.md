# Module 11: Advanced Python Concepts - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master advanced Python programming concepts and techniques
- Implement design patterns and architectural patterns
- Work with metaclasses and advanced OOP concepts
- Understand concurrency and asynchronous programming
- Master memory management and performance optimization
- Build robust, scalable, and maintainable applications
- Apply advanced Python features in real-world scenarios

## Core Concepts

### 1. Metaclasses and Advanced OOP

#### Understanding Metaclasses
```python
# Basic metaclass example
class SingletonMeta(type):
    """Metaclass for creating singleton classes"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton database connection class"""
    def __init__(self):
        self.connection_id = id(self)
        print(f"Database connection created: {self.connection_id}")

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(f"Same instance: {db1 is db2}")  # True

# Advanced metaclass with validation
class ValidatedMeta(type):
    """Metaclass that validates class attributes"""
    
    def __new__(mcs, name, bases, namespace):
        # Validate that required attributes exist
        if 'required_attrs' in namespace:
            required = namespace['required_attrs']
            for attr in required:
                if attr not in namespace:
                    raise AttributeError(f"Required attribute '{attr}' not found")
        
        # Validate attribute types
        if 'attr_types' in namespace:
            attr_types = namespace['attr_types']
            for attr, expected_type in attr_types.items():
                if attr in namespace:
                    if not isinstance(namespace[attr], expected_type):
                        raise TypeError(f"Attribute '{attr}' must be of type {expected_type}")
        
        return super().__new__(mcs, name, bases, namespace)

class ValidatedClass(metaclass=ValidatedMeta):
    required_attrs = ['name', 'value']
    attr_types = {'name': str, 'value': int}
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
```

#### Descriptors and Property Management
```python
class Descriptor:
    """Base descriptor class"""
    
    def __init__(self, name=None):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
    
    def __delete__(self, instance):
        del instance.__dict__[self.name]

class TypedDescriptor(Descriptor):
    """Descriptor with type checking"""
    
    def __init__(self, name, expected_type):
        super().__init__(name)
        self.expected_type = expected_type
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}, got {type(value)}")
        super().__set__(instance, value)

class RangeDescriptor(Descriptor):
    """Descriptor with range validation"""
    
    def __init__(self, name, min_val, max_val):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val
    
    def __set__(self, instance, value):
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value must be between {self.min_val} and {self.max_val}")
        super().__set__(instance, value)

class Person:
    """Example class using descriptors"""
    name = TypedDescriptor('name', str)
    age = RangeDescriptor('age', 0, 150)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Usage
person = Person("Alice", 30)
print(f"Name: {person.name}, Age: {person.age}")

# This will raise TypeError
# person.name = 123

# This will raise ValueError
# person.age = 200
```

### 2. Design Patterns

#### Creational Patterns
```python
# Factory Pattern
class AnimalFactory:
    """Factory for creating different types of animals"""
    
    @staticmethod
    def create_animal(animal_type, name):
        if animal_type == "dog":
            return Dog(name)
        elif animal_type == "cat":
            return Cat(name)
        elif animal_type == "bird":
            return Bird(name)
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

class Bird(Animal):
    def speak(self):
        return f"{self.name} says Tweet!"

# Abstract Factory Pattern
class AbstractFactory:
    """Abstract factory for creating UI components"""
    
    def create_button(self):
        raise NotImplementedError
    
    def create_dialog(self):
        raise NotImplementedError

class WindowsFactory(AbstractFactory):
    def create_button(self):
        return WindowsButton()
    
    def create_dialog(self):
        return WindowsDialog()

class MacFactory(AbstractFactory):
    def create_button(self):
        return MacButton()
    
    def create_dialog(self):
        return MacDialog()

class Button:
    def render(self):
        raise NotImplementedError

class WindowsButton(Button):
    def render(self):
        return "Windows Button"

class MacButton(Button):
    def render(self):
        return "Mac Button"

class Dialog:
    def render(self):
        raise NotImplementedError

class WindowsDialog(Dialog):
    def render(self):
        return "Windows Dialog"

class MacDialog(Dialog):
    def render(self):
        return "Mac Dialog"

# Builder Pattern
class ComputerBuilder:
    """Builder for creating computers"""
    
    def __init__(self):
        self.computer = Computer()
    
    def set_cpu(self, cpu):
        self.computer.cpu = cpu
        return self
    
    def set_memory(self, memory):
        self.computer.memory = memory
        return self
    
    def set_storage(self, storage):
        self.computer.storage = storage
        return self
    
    def set_gpu(self, gpu):
        self.computer.gpu = gpu
        return self
    
    def build(self):
        return self.computer

class Computer:
    def __init__(self):
        self.cpu = None
        self.memory = None
        self.storage = None
        self.gpu = None
    
    def __str__(self):
        return f"Computer: CPU={self.cpu}, Memory={self.memory}, Storage={self.storage}, GPU={self.gpu}"

# Usage
computer = (ComputerBuilder()
           .set_cpu("Intel i7")
           .set_memory("16GB")
           .set_storage("512GB SSD")
           .set_gpu("RTX 3080")
           .build())
print(computer)
```

#### Structural Patterns
```python
# Adapter Pattern
class OldPrinter:
    """Old printer interface"""
    def print_old(self, text):
        return f"Old printer: {text}"

class NewPrinter:
    """New printer interface"""
    def print_new(self, text):
        return f"New printer: {text}"

class PrinterAdapter:
    """Adapter to make old printer work with new interface"""
    
    def __init__(self, old_printer):
        self.old_printer = old_printer
    
    def print_new(self, text):
        return self.old_printer.print_old(text)

# Decorator Pattern
class Coffee:
    """Base coffee class"""
    def cost(self):
        return 2.0
    
    def description(self):
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    """Base decorator class"""
    def __init__(self, coffee):
        self.coffee = coffee
    
    def cost(self):
        return self.coffee.cost()
    
    def description(self):
        return self.coffee.description()

class Milk(CoffeeDecorator):
    def cost(self):
        return self.coffee.cost() + 0.5
    
    def description(self):
        return self.coffee.description() + ", milk"

class Sugar(CoffeeDecorator):
    def cost(self):
        return self.coffee.cost() + 0.2
    
    def description(self):
        return self.coffee.description() + ", sugar"

# Usage
coffee = Coffee()
coffee = Milk(coffee)
coffee = Sugar(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")

# Facade Pattern
class SubsystemA:
    def operation_a(self):
        return "Subsystem A operation"

class SubsystemB:
    def operation_b(self):
        return "Subsystem B operation"

class SubsystemC:
    def operation_c(self):
        return "Subsystem C operation"

class Facade:
    """Facade that simplifies the interface to subsystems"""
    
    def __init__(self):
        self.subsystem_a = SubsystemA()
        self.subsystem_b = SubsystemB()
        self.subsystem_c = SubsystemC()
    
    def operation(self):
        result = []
        result.append(self.subsystem_a.operation_a())
        result.append(self.subsystem_b.operation_b())
        result.append(self.subsystem_c.operation_c())
        return " ".join(result)

# Usage
facade = Facade()
print(facade.operation())
```

#### Behavioral Patterns
```python
# Observer Pattern
class Subject:
    """Subject that notifies observers of changes"""
    
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    def set_state(self, state):
        self._state = state
        self.notify()
    
    def get_state(self):
        return self._state

class Observer:
    """Observer interface"""
    def update(self, subject):
        raise NotImplementedError

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        print(f"{self.name} received update: {subject.get_state()}")

# Strategy Pattern
class PaymentStrategy:
    """Strategy interface for payment methods"""
    def pay(self, amount):
        raise NotImplementedError

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} using credit card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} using PayPal"

class BankTransferPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} using bank transfer"

class PaymentProcessor:
    """Context that uses payment strategies"""
    
    def __init__(self, payment_strategy):
        self.payment_strategy = payment_strategy
    
    def process_payment(self, amount):
        return self.payment_strategy.pay(amount)
    
    def set_payment_strategy(self, payment_strategy):
        self.payment_strategy = payment_strategy

# Command Pattern
class Command:
    """Command interface"""
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError

class Light:
    """Receiver class"""
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        print("Light is on")
    
    def turn_off(self):
        self.is_on = False
        print("Light is off")

class LightOnCommand(Command):
    """Concrete command for turning light on"""
    
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    """Concrete command for turning light off"""
    
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        self.light.turn_off()
    
    def undo(self):
        self.light.turn_on()

class RemoteControl:
    """Invoker class"""
    
    def __init__(self):
        self.commands = {}
        self.last_command = None
    
    def set_command(self, slot, command):
        self.commands[slot] = command
    
    def press_button(self, slot):
        if slot in self.commands:
            self.commands[slot].execute()
            self.last_command = self.commands[slot]
    
    def press_undo(self):
        if self.last_command:
            self.last_command.undo()
```

### 3. Concurrency and Asynchronous Programming

#### Threading
```python
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThreadSafeCounter:
    """Thread-safe counter using locks"""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def decrement(self):
        with self._lock:
            self._value -= 1
    
    def get_value(self):
        with self._lock:
            return self._value

def worker_function(counter, iterations):
    """Worker function that increments counter"""
    for _ in range(iterations):
        counter.increment()
        time.sleep(0.001)  # Simulate work

# Thread pool example
def process_data(data):
    """Process data in a thread"""
    time.sleep(0.1)  # Simulate processing
    return f"Processed: {data}"

def thread_pool_example():
    """Example of using thread pool"""
    data = [f"item_{i}" for i in range(10)]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        futures = [executor.submit(process_data, item) for item in data]
        
        # Collect results
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results

# Producer-Consumer pattern
class ProducerConsumer:
    """Producer-Consumer pattern using queues"""
    
    def __init__(self, max_size=10):
        self.queue = queue.Queue(maxsize=max_size)
        self.stop_event = threading.Event()
    
    def producer(self, name):
        """Producer function"""
        for i in range(10):
            if self.stop_event.is_set():
                break
            
            item = f"{name}_item_{i}"
            self.queue.put(item)
            print(f"Producer {name} produced: {item}")
            time.sleep(0.1)
    
    def consumer(self, name):
        """Consumer function"""
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                print(f"Consumer {name} consumed: {item}")
                self.queue.task_done()
                time.sleep(0.1)
            except queue.Empty:
                continue
    
    def run(self):
        """Run producer-consumer example"""
        # Create threads
        producer_threads = [
            threading.Thread(target=self.producer, args=(f"P{i}",))
            for i in range(2)
        ]
        
        consumer_threads = [
            threading.Thread(target=self.consumer, args=(f"C{i}",))
            for i in range(3)
        ]
        
        # Start threads
        for thread in producer_threads + consumer_threads:
            thread.start()
        
        # Wait for producers to finish
        for thread in producer_threads:
            thread.join()
        
        # Stop consumers
        self.stop_event.set()
        
        # Wait for consumers to finish
        for thread in consumer_threads:
            thread.join()
```

#### Asynchronous Programming
```python
import asyncio
import aiohttp
import time

class AsyncProcessor:
    """Asynchronous data processor"""
    
    def __init__(self):
        self.results = []
    
    async def process_item(self, item, delay=0.1):
        """Process a single item asynchronously"""
        await asyncio.sleep(delay)  # Simulate async work
        result = f"Processed: {item}"
        self.results.append(result)
        return result
    
    async def process_batch(self, items, max_concurrent=5):
        """Process multiple items with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_item(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks)
        return results
    
    async def fetch_url(self, session, url):
        """Fetch URL asynchronously"""
        try:
            async with session.get(url) as response:
                return await response.text()
        except Exception as e:
            return f"Error fetching {url}: {e}"
    
    async def fetch_multiple_urls(self, urls):
        """Fetch multiple URLs concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results

# Async context manager
class AsyncDatabaseConnection:
    """Async database connection context manager"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        print(f"Connecting to database: {self.connection_string}")
        # Simulate connection
        await asyncio.sleep(0.1)
        self.connection = f"Connection to {self.connection_string}"
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        # Simulate cleanup
        await asyncio.sleep(0.1)
        self.connection = None

# Async generator
async def async_generator(n):
    """Async generator that yields values"""
    for i in range(n):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i

# Usage example
async def main():
    """Main async function"""
    processor = AsyncProcessor()
    
    # Process items asynchronously
    items = [f"item_{i}" for i in range(10)]
    results = await processor.process_batch(items, max_concurrent=3)
    print(f"Processed {len(results)} items")
    
    # Fetch URLs asynchronously
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]
    
    # Note: This would require aiohttp to be installed
    # url_results = await processor.fetch_multiple_urls(urls)
    # print(f"Fetched {len(url_results)} URLs")
    
    # Use async context manager
    async with AsyncDatabaseConnection("postgresql://localhost/mydb") as conn:
        print(f"Using connection: {conn}")
        await asyncio.sleep(0.1)  # Simulate database work
    
    # Use async generator
    async for value in async_generator(5):
        print(f"Generated value: {value}")

# Run async main
# asyncio.run(main())
```

### 4. Memory Management and Performance

#### Memory Optimization
```python
import sys
import gc
import weakref
from functools import lru_cache
import tracemalloc

class MemoryOptimizedClass:
    """Class optimized for memory usage"""
    
    __slots__ = ['name', 'value', 'data']  # Restrict attributes
    
    def __init__(self, name, value, data=None):
        self.name = name
        self.value = value
        self.data = data or []
    
    def __del__(self):
        print(f"Deleting {self.name}")

class MemoryManager:
    """Memory management utilities"""
    
    def __init__(self):
        self.objects = []
        self.weak_refs = []
    
    def add_object(self, obj):
        """Add object with weak reference"""
        self.objects.append(obj)
        self.weak_refs.append(weakref.ref(obj))
    
    def get_memory_usage(self):
        """Get current memory usage"""
        return sys.getsizeof(self.objects)
    
    def cleanup_dead_refs(self):
        """Clean up dead weak references"""
        alive_refs = []
        for ref in self.weak_refs:
            if ref() is not None:
                alive_refs.append(ref)
        self.weak_refs = alive_refs
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        collected = gc.collect()
        print(f"Collected {collected} objects")
        return collected

# Memory profiling
class MemoryProfiler:
    """Memory profiler for tracking memory usage"""
    
    def __init__(self):
        self.snapshots = []
        tracemalloc.start()
    
    def take_snapshot(self, label=""):
        """Take memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
        return snapshot
    
    def compare_snapshots(self, label1, label2):
        """Compare two snapshots"""
        snap1 = next(s for l, s in self.snapshots if l == label1)
        snap2 = next(s for l, s in self.snapshots if l == label2)
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        return top_stats
    
    def get_current_memory(self):
        """Get current memory usage"""
        current, peak = tracemalloc.get_traced_memory()
        return current, peak

# Performance optimization with caching
class OptimizedCalculator:
    """Calculator with caching for performance"""
    
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def fibonacci(self, n):
        """Calculate Fibonacci number with caching"""
        if n < 2:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)
    
    def expensive_calculation(self, x, y):
        """Expensive calculation with manual caching"""
        key = (x, y)
        if key in self._cache:
            return self._cache[key]
        
        # Simulate expensive calculation
        result = sum(i * j for i in range(x) for j in range(y))
        self._cache[key] = result
        return result
    
    def clear_cache(self):
        """Clear calculation cache"""
        self._cache.clear()
        self.fibonacci.cache_clear()

# Memory-efficient data structures
class MemoryEfficientList:
    """Memory-efficient list implementation"""
    
    def __init__(self, initial_capacity=10):
        self._capacity = initial_capacity
        self._size = 0
        self._data = [None] * initial_capacity
    
    def append(self, item):
        """Append item to list"""
        if self._size >= self._capacity:
            self._resize()
        self._data[self._size] = item
        self._size += 1
    
    def _resize(self):
        """Resize internal array"""
        self._capacity *= 2
        new_data = [None] * self._capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
    
    def __getitem__(self, index):
        """Get item by index"""
        if 0 <= index < self._size:
            return self._data[index]
        raise IndexError("Index out of range")
    
    def __len__(self):
        """Get list length"""
        return self._size
    
    def __str__(self):
        """String representation"""
        return str(self._data[:self._size])
```

### 5. Advanced Python Features

#### Context Managers
```python
from contextlib import contextmanager, ExitStack
import os
import tempfile

class DatabaseConnection:
    """Database connection context manager"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = f"Connection to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        self.connection = None
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

@contextmanager
def temporary_file(content=""):
    """Context manager for temporary file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    try:
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
    finally:
        os.unlink(temp_file.name)

@contextmanager
def change_directory(path):
    """Context manager for changing directory"""
    original_path = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_path)

# Multiple context managers
def multiple_contexts_example():
    """Example of using multiple context managers"""
    with ExitStack() as stack:
        # Enter multiple contexts
        db_conn = stack.enter_context(DatabaseConnection("postgresql://localhost/db"))
        temp_file = stack.enter_context(temporary_file("Hello, World!"))
        stack.enter_context(change_directory("/tmp"))
        
        print(f"Using database: {db_conn}")
        print(f"Using temp file: {temp_file}")
        print(f"Current directory: {os.getcwd()}")
        
        # All contexts will be exited automatically
```

#### Generators and Iterators
```python
class FibonacciGenerator:
    """Generator for Fibonacci numbers"""
    
    def __init__(self, max_count=None):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.max_count and self.count >= self.max_count:
            raise StopIteration
        
        if self.count == 0:
            self.count += 1
            return self.a
        elif self.count == 1:
            self.count += 1
            return self.b
        else:
            self.a, self.b = self.b, self.a + self.b
            self.count += 1
            return self.b

def fibonacci_generator(max_count=None):
    """Generator function for Fibonacci numbers"""
    a, b = 0, 1
    count = 0
    
    while max_count is None or count < max_count:
        yield a
        a, b = b, a + b
        count += 1

class DataProcessor:
    """Data processor with generator methods"""
    
    def __init__(self, data):
        self.data = data
    
    def filter_data(self, predicate):
        """Filter data using generator"""
        for item in self.data:
            if predicate(item):
                yield item
    
    def transform_data(self, transformer):
        """Transform data using generator"""
        for item in self.data:
            yield transformer(item)
    
    def batch_data(self, batch_size):
        """Batch data into chunks"""
        batch = []
        for item in self.data:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining items
            yield batch

# Generator expressions and comprehensions
def generator_examples():
    """Examples of generator expressions"""
    # Generator expression
    squares = (x**2 for x in range(10))
    print(f"Squares: {list(squares)}")
    
    # Generator with condition
    even_squares = (x**2 for x in range(10) if x % 2 == 0)
    print(f"Even squares: {list(even_squares)}")
    
    # Generator with multiple iterables
    pairs = ((x, y) for x in range(3) for y in range(3))
    print(f"Pairs: {list(pairs)}")
    
    # Generator with function
    def square(x):
        return x**2
    
    squares_func = (square(x) for x in range(5))
    print(f"Squares with function: {list(squares_func)}")

# Coroutine example
async def async_generator(n):
    """Async generator example"""
    for i in range(n):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i

async def process_async_generator():
    """Process async generator"""
    async for value in async_generator(5):
        print(f"Async generated value: {value}")
```

#### Advanced Decorators
```python
import functools
import time
import logging
from typing import Callable, Any

def retry(max_attempts=3, delay=1, backoff=2):
    """Decorator for retrying failed functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def rate_limit(calls_per_second=1):
    """Decorator for rate limiting function calls"""
    def decorator(func):
        last_called = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 1.0 / calls_per_second - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def validate_types(**type_map):
    """Decorator for validating function argument types"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = functools.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_map.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(f"Parameter '{param_name}' must be of type {expected_type}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache_result(ttl_seconds=300):
    """Decorator for caching function results with TTL"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result is still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: f"Cache size: {len(cache)}"
        
        return wrapper
    return decorator

def log_execution(logger=None):
    """Decorator for logging function execution"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        return wrapper
    return decorator

# Example usage
@retry(max_attempts=3, delay=1)
@rate_limit(calls_per_second=2)
@validate_types(x=int, y=str)
@cache_result(ttl_seconds=60)
@log_execution()
def example_function(x, y="default"):
    """Example function with multiple decorators"""
    if x < 0:
        raise ValueError("x must be positive")
    return f"Result: {x} + {y}"

# Class decorator
def singleton(cls):
    """Class decorator for singleton pattern"""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseManager:
    """Singleton database manager"""
    def __init__(self):
        self.connection = None
        print("DatabaseManager instance created")
    
    def connect(self, connection_string):
        self.connection = connection_string
        print(f"Connected to {connection_string}")

# Usage
db1 = DatabaseManager()
db2 = DatabaseManager()
print(f"Same instance: {db1 is db2}")  # True
```

## Best Practices

### 1. Code Organization and Architecture
```python
# Package structure example
"""
my_package/
    __init__.py
    core/
        __init__.py
        base.py
        interfaces.py
    services/
        __init__.py
        database.py
        api.py
    utils/
        __init__.py
        helpers.py
        validators.py
    tests/
        __init__.py
        test_core.py
        test_services.py
"""

# Interface definition
from abc import ABC, abstractmethod

class DataRepository(ABC):
    """Abstract base class for data repositories"""
    
    @abstractmethod
    def save(self, data):
        """Save data to repository"""
        pass
    
    @abstractmethod
    def find_by_id(self, id):
        """Find data by ID"""
        pass
    
    @abstractmethod
    def delete(self, id):
        """Delete data by ID"""
        pass

class DatabaseRepository(DataRepository):
    """Database implementation of data repository"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def save(self, data):
        print(f"Saving to database: {data}")
        return True
    
    def find_by_id(self, id):
        print(f"Finding in database by ID: {id}")
        return f"Data with ID {id}"
    
    def delete(self, id):
        print(f"Deleting from database by ID: {id}")
        return True

# Dependency injection
class ServiceContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, interface, implementation, singleton=False):
        """Register service implementation"""
        self._services[interface] = (implementation, singleton)
    
    def get(self, interface):
        """Get service instance"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        implementation, is_singleton = self._services[interface]
        
        if is_singleton:
            if interface not in self._singletons:
                self._singletons[interface] = implementation()
            return self._singletons[interface]
        else:
            return implementation()

# Usage
container = ServiceContainer()
container.register(DataRepository, DatabaseRepository, singleton=True)

repository = container.get(DataRepository)
repository.save("test data")
```

### 2. Error Handling and Logging
```python
import logging
import traceback
from typing import Optional, Callable, Any

class CustomException(Exception):
    """Custom exception with additional context"""
    
    def __init__(self, message, error_code=None, details=None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.handlers = {}
    
    def register_handler(self, exception_type, handler):
        """Register exception handler"""
        self.handlers[exception_type] = handler
    
    def handle(self, exception, context=None):
        """Handle exception"""
        exception_type = type(exception)
        
        if exception_type in self.handlers:
            return self.handlers[exception_type](exception, context)
        else:
            self.logger.error(f"Unhandled exception: {exception}", exc_info=True)
            return None
    
    def safe_execute(self, func, *args, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle(e, {"function": func.__name__, "args": args, "kwargs": kwargs})

# Logging configuration
def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

# Usage
setup_logging()
error_handler = ErrorHandler()

def risky_function(x):
    if x < 0:
        raise CustomException("Negative value not allowed", error_code="NEGATIVE_VALUE")
    return x * 2

result = error_handler.safe_execute(risky_function, -5)
print(f"Result: {result}")
```

## Quick Checks

### Check 1: Metaclasses
```python
# What will this print?
class Meta(type):
    def __new__(cls, name, bases, namespace):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass
```

### Check 2: Descriptors
```python
# What will this do?
class Descriptor:
    def __get__(self, instance, owner):
        return "Descriptor value"
    
    def __set__(self, instance, value):
        print(f"Setting value: {value}")

class Test:
    attr = Descriptor()

obj = Test()
print(obj.attr)
obj.attr = "new value"
```

### Check 3: Generators
```python
# What will this output?
def generator():
    yield 1
    yield 2
    yield 3

gen = generator()
print(next(gen))
print(next(gen))
print(list(gen))
```

## Lab Problems

### Lab 1: Design Pattern Implementation
Implement a comprehensive system using multiple design patterns (Factory, Observer, Strategy, Command).

### Lab 2: Asynchronous Application
Build an asynchronous web scraper or API client using asyncio and aiohttp.

### Lab 3: Memory-Optimized Data Structure
Create a memory-efficient data structure for handling large datasets.

### Lab 4: Plugin System
Design and implement a plugin system using metaclasses and dynamic loading.

## AI Code Comparison
When working with AI-generated advanced Python code, evaluate:
- **Design patterns** - are the patterns correctly implemented and appropriate?
- **Performance** - is the code optimized for the specific use case?
- **Memory usage** - are there memory leaks or inefficient memory usage?
- **Concurrency** - is the concurrent code thread-safe and efficient?
- **Error handling** - are exceptions properly handled and logged?

## Next Steps
- Learn about Python C extensions and Cython
- Master advanced debugging and profiling techniques
- Explore Python packaging and distribution
- Study Python security best practices
- Understand Python in cloud and containerized environments
