# Module 13: Advanced Python Concepts - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master advanced Python features like metaclasses, descriptors, and decorators
- Implement design patterns and architectural patterns in Python
- Work with asynchronous programming using asyncio and async/await
- Understand Python's memory management and performance optimization
- Implement advanced data structures and algorithms
- Master Python's introspection and metaprogramming capabilities
- Build high-performance Python applications

## Core Concepts

### 1. Metaclasses and Class Creation
Metaclasses are classes that create other classes. They allow you to customize class creation behavior.

```python
# Basic metaclass example
class SingletonMeta(type):
    """Metaclass that creates singleton classes"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Database connection established"
    
    def query(self, sql):
        return f"Executing: {sql}"

# Both instances will be the same
db1 = Database()
db2 = Database()
print(db1 is db2)  # True

# Advanced metaclass with validation
class ValidatedMeta(type):
    """Metaclass that validates class attributes"""
    
    def __new__(mcs, name, bases, namespace):
        # Validate required attributes
        if 'required_attrs' in namespace:
            required = namespace['required_attrs']
            for attr in required:
                if attr not in namespace:
                    raise TypeError(f"Class {name} must define {attr}")
        
        # Add validation methods
        namespace['_validate_attributes'] = mcs._validate_attributes
        
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def _validate_attributes(self):
        """Validate instance attributes"""
        for attr in getattr(self.__class__, 'required_attrs', []):
            if not hasattr(self, attr):
                raise AttributeError(f"Instance must have {attr}")

class Model(metaclass=ValidatedMeta):
    required_attrs = ['name', 'value']
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self._validate_attributes()
```

### 2. Descriptors and Property Management
Descriptors allow you to customize attribute access and provide powerful property management.

```python
class TypedProperty:
    """Descriptor that enforces type checking"""
    
    def __init__(self, expected_type, default=None):
        self.expected_type = expected_type
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be of type {self.expected_type}")
        instance.__dict__[self.name] = value

class CachedProperty:
    """Descriptor that caches property values"""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        if self.name not in instance.__dict__:
            instance.__dict__[self.name] = self.func(instance)
        return instance.__dict__[self.name]

class Person:
    name = TypedProperty(str)
    age = TypedProperty(int, default=0)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @CachedProperty
    def full_info(self):
        """Expensive computation that gets cached"""
        import time
        time.sleep(1)  # Simulate expensive operation
        return f"{self.name} is {self.age} years old"

# Usage
person = Person("Alice", 30)
print(person.full_info)  # First call - slow
print(person.full_info)  # Second call - fast (cached)
```

### 3. Advanced Decorators
Decorators can be used for much more than simple function wrapping.

```python
import functools
import time
import logging
from typing import Callable, Any

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator that retries a function on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def rate_limit(calls_per_second: float):
    """Decorator that rate limits function calls"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            time_since_last = now - last_called[0]
            
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_types(**type_hints):
    """Decorator that validates function argument types"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{param_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success!"

@rate_limit(calls_per_second=2.0)
def api_call():
    print("Making API call...")
    return "API response"

@validate_types(name=str, age=int, active=bool)
def create_user(name, age, active=True):
    return {"name": name, "age": age, "active": active}
```

### 4. Asynchronous Programming
Python's asyncio provides powerful tools for concurrent programming.

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

class AsyncDataProcessor:
    """Asynchronous data processor"""
    
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Fetch data from URL with rate limiting"""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"url": url, "data": data, "status": "success"}
                    else:
                        return {"url": url, "error": f"HTTP {response.status}", "status": "error"}
            except Exception as e:
                return {"url": url, "error": str(e), "status": "error"}
    
    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs concurrently"""
        tasks = [self.fetch_data(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def batch_process(self, items: List[Any], batch_size: int = 10) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [self.process_item(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def process_item(self, item: Any) -> Any:
        """Process a single item (simulate async work)"""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Processed: {item}"

# Async context manager
class AsyncDatabase:
    """Async database connection manager"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        # Simulate async connection
        await asyncio.sleep(0.1)
        self.connection = f"Connected to {self.connection_string}"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            await asyncio.sleep(0.1)  # Simulate async cleanup
            self.connection = None
    
    async def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute async query"""
        await asyncio.sleep(0.1)  # Simulate async query
        return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

# Usage example
async def main():
    urls = [
        "https://api.github.com/users/octocat",
        "https://api.github.com/users/torvalds",
        "https://api.github.com/users/gvanrossum"
    ]
    
    async with AsyncDataProcessor(max_concurrent=5) as processor:
        results = await processor.process_urls(urls)
        for result in results:
            print(f"URL: {result['url']}, Status: {result['status']}")
    
    # Batch processing
    items = list(range(20))
    processed = await processor.batch_process(items, batch_size=5)
    print(f"Processed {len(processed)} items")
    
    # Database operations
    async with AsyncDatabase("postgresql://localhost/mydb") as db:
        users = await db.query("SELECT * FROM users")
        print(f"Found {len(users)} users")

# Run async main
if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Memory Management and Performance
Understanding Python's memory management is crucial for building high-performance applications.

```python
import sys
import gc
import weakref
from memory_profiler import profile
import tracemalloc

class MemoryTracker:
    """Track memory usage of objects"""
    
    def __init__(self):
        self.objects = weakref.WeakSet()
        self.tracemalloc_started = False
    
    def start_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
        self.tracemalloc_started = True
    
    def stop_tracing(self):
        """Stop memory tracing and get statistics"""
        if self.tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return current, peak
        return None, None
    
    def track_object(self, obj):
        """Track an object for memory monitoring"""
        self.objects.add(obj)
        return obj
    
    def get_memory_usage(self):
        """Get current memory usage"""
        return sys.getsizeof(self.objects)

class OptimizedDataProcessor:
    """Memory-optimized data processor"""
    
    def __init__(self):
        self.memory_tracker = MemoryTracker()
        self.memory_tracker.start_tracing()
    
    def process_large_dataset(self, data):
        """Process large dataset with memory optimization"""
        # Use generator to avoid loading all data into memory
        for chunk in self._chunk_data(data, chunk_size=1000):
            processed_chunk = self._process_chunk(chunk)
            yield processed_chunk
            # Force garbage collection after each chunk
            gc.collect()
    
    def _chunk_data(self, data, chunk_size):
        """Split data into chunks"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    def _process_chunk(self, chunk):
        """Process a single chunk"""
        # Simulate processing
        return [item * 2 for item in chunk]
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        current, peak = self.memory_tracker.stop_tracing()
        if current and peak:
            print(f"Memory usage - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")

# Memory profiling decorator
def memory_profile(func):
    """Decorator to profile memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Function {func.__name__} - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
        return result
    return wrapper

@memory_profile
def process_large_list():
    """Process a large list with memory profiling"""
    data = list(range(1000000))
    processor = OptimizedDataProcessor()
    results = list(processor.process_large_dataset(data))
    return len(results)
```

### 6. Advanced Data Structures
Implementing custom data structures for specific use cases.

```python
from collections import deque
from typing import Any, Optional, List
import heapq

class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)

class Trie:
    """Trie (Prefix Tree) implementation"""
    
    def __init__(self):
        self.root = {}
        self.end_marker = '*'
    
    def insert(self, word: str) -> None:
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_marker] = True
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_marker in node
    
    def starts_with(self, prefix: str) -> List[str]:
        """Find all words with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node: dict, prefix: str, words: List[str]) -> None:
        """Collect all words from node"""
        for char, child in node.items():
            if char == self.end_marker:
                words.append(prefix)
            else:
                self._collect_words(child, prefix + char, words)

class PriorityQueue:
    """Priority queue implementation using heapq"""
    
    def __init__(self):
        self.heap = []
        self.index = 0
    
    def push(self, item: Any, priority: float) -> None:
        """Push item with priority"""
        heapq.heappush(self.heap, (priority, self.index, item))
        self.index += 1
    
    def pop(self) -> Any:
        """Pop highest priority item"""
        if not self.heap:
            raise IndexError("Priority queue is empty")
        priority, index, item = heapq.heappop(self.heap)
        return item
    
    def peek(self) -> Any:
        """Peek at highest priority item without removing"""
        if not self.heap:
            raise IndexError("Priority queue is empty")
        priority, index, item = self.heap[0]
        return item
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.heap)

# Usage examples
def test_data_structures():
    """Test custom data structures"""
    
    # LRU Cache
    cache = LRUCache(3)
    cache.put(1, "one")
    cache.put(2, "two")
    cache.put(3, "three")
    print(cache.get(1))  # "one"
    cache.put(4, "four")  # Evicts key 2
    print(cache.get(2))  # None
    
    # Trie
    trie = Trie()
    trie.insert("hello")
    trie.insert("world")
    trie.insert("help")
    print(trie.search("hello"))  # True
    print(trie.search("hel"))    # False
    print(trie.starts_with("hel"))  # ["hello", "help"]
    
    # Priority Queue
    pq = PriorityQueue()
    pq.push("task1", 3)
    pq.push("task2", 1)
    pq.push("task3", 2)
    print(pq.pop())  # "task2" (highest priority)
    print(pq.pop())  # "task3"
    print(pq.pop())  # "task1"
```

### 7. Metaprogramming and Introspection
Python's introspection capabilities allow for powerful metaprogramming.

```python
import inspect
import types
from typing import Any, Callable, Dict

class DynamicClass:
    """Class that can be modified at runtime"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def add_method(self, name: str, func: Callable) -> None:
        """Add method to class at runtime"""
        setattr(self, name, types.MethodType(func, self))
    
    def add_property(self, name: str, getter: Callable, setter: Callable = None) -> None:
        """Add property to class at runtime"""
        if setter:
            prop = property(getter, setter)
        else:
            prop = property(getter)
        setattr(self.__class__, name, prop)

def create_class_from_dict(class_name: str, attributes: Dict[str, Any]) -> type:
    """Create a class dynamically from a dictionary"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{class_name}({attrs})"
    
    # Create class dynamically
    new_class = type(class_name, (), {
        '__init__': __init__,
        '__str__': __str__,
        **attributes
    })
    
    return new_class

def monkey_patch_class(cls: type, method_name: str, new_method: Callable) -> None:
    """Monkey patch a method in a class"""
    original_method = getattr(cls, method_name, None)
    
    def wrapper(self, *args, **kwargs):
        return new_method(self, *args, **kwargs)
    
    wrapper.__name__ = method_name
    wrapper.__doc__ = new_method.__doc__
    setattr(cls, method_name, wrapper)
    
    return original_method

def inspect_function(func: Callable) -> Dict[str, Any]:
    """Inspect a function and return metadata"""
    sig = inspect.signature(func)
    
    return {
        'name': func.__name__,
        'docstring': func.__doc__,
        'annotations': func.__annotations__,
        'parameters': {
            name: {
                'annotation': param.annotation,
                'default': param.default,
                'kind': param.kind
            }
            for name, param in sig.parameters.items()
        },
        'return_annotation': sig.return_annotation,
        'is_async': inspect.iscoroutinefunction(func),
        'is_generator': inspect.isgeneratorfunction(func)
    }

# Usage examples
def test_metaprogramming():
    """Test metaprogramming features"""
    
    # Dynamic class creation
    Person = create_class_from_dict('Person', {
        'greet': lambda self: f"Hello, I'm {getattr(self, 'name', 'Unknown')}"
    })
    
    person = Person(name="Alice", age=30)
    print(person)  # Person(name=Alice, age=30)
    print(person.greet())  # Hello, I'm Alice
    
    # Dynamic object modification
    obj = DynamicClass(name="Bob", age=25)
    
    def say_hello(self):
        return f"Hi, I'm {self.name}"
    
    obj.add_method('say_hello', say_hello)
    print(obj.say_hello())  # Hi, I'm Bob
    
    # Function inspection
    def example_func(name: str, age: int = 25, *args, **kwargs) -> str:
        """Example function for inspection"""
        return f"Name: {name}, Age: {age}"
    
    metadata = inspect_function(example_func)
    print(metadata)
```

## Best Practices

### 1. Performance Optimization
```python
import cProfile
import pstats
from functools import lru_cache
import numpy as np

class PerformanceProfiler:
    """Profile and optimize Python code"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function and return statistics"""
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        return {
            'result': result,
            'total_calls': stats.total_calls,
            'total_time': stats.total_tt,
            'stats': stats
        }
    
    def optimize_with_caching(self, func: Callable) -> Callable:
        """Add caching to a function"""
        return lru_cache(maxsize=128)(func)
    
    def vectorize_operation(self, data: List[float]) -> np.ndarray:
        """Use NumPy for vectorized operations"""
        arr = np.array(data)
        return arr * 2  # Vectorized multiplication

# Usage
profiler = PerformanceProfiler()

@profiler.optimize_with_caching
def fibonacci(n: int) -> int:
    """Fibonacci with caching"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Profile the function
result = profiler.profile_function(fibonacci, 30)
print(f"Fibonacci(30) = {result['result']}")
print(f"Total calls: {result['total_calls']}")
```

### 2. Error Handling and Debugging
```python
import traceback
import logging
from contextlib import contextmanager
from typing import Optional, Any

class AdvancedErrorHandler:
    """Advanced error handling and debugging"""
    
    def __init__(self, log_file: str = "debug.log"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    @contextmanager
    def debug_context(self, operation_name: str):
        """Context manager for debugging operations"""
        self.logger.info(f"Starting operation: {operation_name}")
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(f"Completed operation: {operation_name} in {duration:.2f}s")
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
```

## Quick Checks

### Check 1: Metaclasses
```python
# What will this print?
class Meta(type):
    def __new__(cls, name, bases, namespace):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass
```

### Check 2: Descriptors
```python
# What will this print?
class Descriptor:
    def __get__(self, instance, owner):
        return "Descriptor value"

class MyClass:
    attr = Descriptor()

obj = MyClass()
print(obj.attr)
```

### Check 3: Async Programming
```python
# What will this print?
import asyncio

async def async_func():
    await asyncio.sleep(1)
    return "Done"

async def main():
    result = await async_func()
    print(result)

asyncio.run(main())
```

## Lab Problems

### Lab 1: Custom ORM
Build a custom Object-Relational Mapping system using metaclasses and descriptors.

### Lab 2: Async Web Scraper
Create an asynchronous web scraper that can handle thousands of URLs efficiently.

### Lab 3: Memory Profiler
Implement a memory profiler that can track and analyze memory usage in Python applications.

### Lab 4: Plugin System
Build a plugin system that allows dynamic loading and execution of Python modules.

## AI Code Comparison
When working with AI-generated advanced Python code, evaluate:
- **Performance implications** - is the code optimized for speed and memory usage?
- **Error handling** - are edge cases and exceptions properly handled?
- **Code complexity** - is the code unnecessarily complex or could it be simplified?
- **Best practices** - does the code follow Python idioms and best practices?
- **Maintainability** - is the code readable and maintainable?

## Next Steps
- Learn about Python's C extensions and Cython
- Master advanced debugging and profiling techniques
- Explore Python's multiprocessing and threading capabilities
- Study Python's packaging and distribution ecosystem
- Understand Python's role in scientific computing and data analysis
