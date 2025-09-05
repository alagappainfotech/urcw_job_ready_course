"""
Module 11: Advanced Python Concepts - Exercises
Complete these exercises to master advanced Python concepts.
"""

import asyncio
import threading
import time
import weakref
import gc
import sys
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
import queue
import tracemalloc

# Exercise 1: Metaclasses and Advanced OOP
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
        self.connected = False
    
    def connect(self, connection_string):
        self.connection_string = connection_string
        self.connected = True
        print(f"Connected to database: {connection_string}")
    
    def disconnect(self):
        self.connected = False
        print("Disconnected from database")

class ValidatedMeta(type):
    """Metaclass that validates class attributes"""
    
    def __new__(mcs, name, bases, namespace):
        # Validate required attributes
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

# Exercise 2: Descriptors and Property Management
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

# Exercise 3: Design Patterns
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

# Exercise 4: Concurrency and Threading
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

# Exercise 5: Asynchronous Programming
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

class AsyncDatabaseConnection:
    """Async database connection context manager"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        print(f"Connecting to database: {self.connection_string}")
        await asyncio.sleep(0.1)  # Simulate connection
        self.connection = f"Connection to {self.connection_string}"
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        await asyncio.sleep(0.1)  # Simulate cleanup
        self.connection = None

async def async_generator(n):
    """Async generator that yields values"""
    for i in range(n):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i

# Exercise 6: Memory Management and Performance
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

class OptimizedCalculator:
    """Calculator with caching for performance"""
    
    def __init__(self):
        self._cache = {}
    
    @functools.lru_cache(maxsize=128)
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

# Exercise 7: Advanced Decorators
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

def singleton(cls):
    """Class decorator for singleton pattern"""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

# Exercise 8: Context Managers and Generators
class DatabaseConnection:
    """Database connection context manager"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
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
    import tempfile
    import os
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    try:
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
    finally:
        os.unlink(temp_file.name)

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

# Test Functions
def test_exercises():
    """Test all exercises"""
    print("Testing Module 11 Exercises...")
    
    # Test 1: Metaclasses
    print("\n1. Testing Metaclasses:")
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"Same instance: {db1 is db2}")  # True
    
    # Test 2: Descriptors
    print("\n2. Testing Descriptors:")
    person = Person("Alice", 30)
    print(f"Name: {person.name}, Age: {person.age}")
    
    # Test 3: Design Patterns
    print("\n3. Testing Design Patterns:")
    
    # Factory Pattern
    dog = AnimalFactory.create_animal("dog", "Buddy")
    print(dog.speak())
    
    # Observer Pattern
    subject = Subject()
    observer1 = ConcreteObserver("Observer1")
    observer2 = ConcreteObserver("Observer2")
    
    subject.attach(observer1)
    subject.attach(observer2)
    subject.set_state("New state")
    
    # Strategy Pattern
    payment_processor = PaymentProcessor(CreditCardPayment())
    print(payment_processor.process_payment(100))
    
    payment_processor.set_payment_strategy(PayPalPayment())
    print(payment_processor.process_payment(200))
    
    # Test 4: Threading
    print("\n4. Testing Threading:")
    counter = ThreadSafeCounter()
    
    # Create and start threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_function, args=(counter, 100))
        threads.append(thread)
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Final counter value: {counter.get_value()}")
    
    # Test 5: Asynchronous Programming
    print("\n5. Testing Asynchronous Programming:")
    
    async def async_test():
        processor = AsyncProcessor()
        items = [f"item_{i}" for i in range(5)]
        results = await processor.process_batch(items, max_concurrent=3)
        print(f"Processed {len(results)} items asynchronously")
        
        # Test async context manager
        async with AsyncDatabaseConnection("postgresql://localhost/db") as conn:
            print(f"Using connection: {conn}")
            await asyncio.sleep(0.1)
        
        # Test async generator
        async for value in async_generator(3):
            print(f"Async generated value: {value}")
    
    # Run async test
    asyncio.run(async_test())
    
    # Test 6: Memory Management
    print("\n6. Testing Memory Management:")
    profiler = MemoryProfiler()
    
    # Take initial snapshot
    profiler.take_snapshot("initial")
    
    # Create some objects
    objects = [MemoryOptimizedClass(f"obj_{i}", i) for i in range(1000)]
    
    # Take snapshot after creating objects
    profiler.take_snapshot("after_creation")
    
    # Get memory usage
    current, peak = profiler.get_current_memory()
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    # Test 7: Advanced Decorators
    print("\n7. Testing Advanced Decorators:")
    
    @retry(max_attempts=3, delay=0.1)
    @rate_limit(calls_per_second=5)
    @validate_types(x=int, y=str)
    @cache_result(ttl_seconds=60)
    def example_function(x, y="default"):
        if x < 0:
            raise ValueError("x must be positive")
        return f"Result: {x} + {y}"
    
    # Test the decorated function
    try:
        result = example_function(5, "test")
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function error: {e}")
    
    # Test 8: Context Managers and Generators
    print("\n8. Testing Context Managers and Generators:")
    
    # Test context manager
    with DatabaseConnection("postgresql://localhost/mydb") as conn:
        print(f"Using connection: {conn}")
    
    # Test generator
    fib_gen = FibonacciGenerator(10)
    fib_numbers = list(fib_gen)
    print(f"Fibonacci numbers: {fib_numbers}")
    
    # Test data processor
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    processor = DataProcessor(data)
    
    # Filter even numbers
    even_numbers = list(processor.filter_data(lambda x: x % 2 == 0))
    print(f"Even numbers: {even_numbers}")
    
    # Transform data
    squared_numbers = list(processor.transform_data(lambda x: x ** 2))
    print(f"Squared numbers: {squared_numbers}")
    
    # Batch data
    batches = list(processor.batch_data(3))
    print(f"Batches: {batches}")
    
    print("\nAll exercises completed!")

if __name__ == "__main__":
    test_exercises()
