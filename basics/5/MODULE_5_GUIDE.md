# Module 5: Modules and Packages - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Understand Python's module system and import mechanisms
- Create and organize modules effectively
- Build packages with proper structure
- Master the `__init__.py` file and `__all__` attribute
- Understand the module search path (`sys.path`)
- Use virtual environments for dependency management
- Work with third-party packages via `pip`

## Core Concepts

### 1. What is a Module?
A module is a file containing Python definitions and statements. The file name is the module name with the suffix `.py` added.

```python
# math_utils.py (this becomes the 'math_utils' module)
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
```

### 2. Import Statements
```python
# Import entire module
import math_utils

# Import specific functions
from math_utils import add, multiply

# Import with alias
import math_utils as mu

# Import all (not recommended)
from math_utils import *
```

### 3. Module Search Path
Python searches for modules in the following order:
1. Current directory
2. Directories in `PYTHONPATH` environment variable
3. Standard library directories
4. Site-packages directory

```python
import sys
print(sys.path)  # Shows the search path
```

### 4. The `__init__.py` File
- Makes a directory a Python package
- Can contain package initialization code
- Controls what gets imported with `from package import *`

```python
# __init__.py
from .module1 import function1
from .module2 import function2

__all__ = ['function1', 'function2']
```

### 5. The `__all__` Attribute
Controls what gets imported when using `from package import *`:

```python
# __init__.py
__all__ = ['public_function', 'PublicClass']

def public_function():
    pass

def _private_function():
    pass

class PublicClass:
    pass

class _PrivateClass:
    pass
```

### 6. Package Structure
```
my_package/
├── __init__.py
├── module1.py
├── module2.py
├── subpackage/
│   ├── __init__.py
│   ├── submodule1.py
│   └── submodule2.py
└── tests/
    ├── __init__.py
    └── test_module1.py
```

## Advanced Topics

### 1. Relative Imports
```python
# Within a package
from .module1 import function1  # Same level
from ..parent_package import function2  # Parent level
from .subpackage.module import function3  # Subpackage
```

### 2. Dynamic Imports
```python
import importlib

# Import module dynamically
module_name = "math"
math_module = importlib.import_module(module_name)

# Import from string
module = importlib.import_module("package.subpackage.module")
```

### 3. Module Reloading
```python
import importlib
import my_module

# Reload module (useful in development)
importlib.reload(my_module)
```

### 4. Package Metadata
```python
# setup.py or pyproject.toml
from setuptools import setup, find_packages

setup(
    name="my_package",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.20.0",
    ],
)
```

## Virtual Environments

### 1. Creating Virtual Environments
```bash
# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Activate (Unix/macOS)
source myenv/bin/activate

# Deactivate
deactivate
```

### 2. Managing Dependencies
```bash
# Install package
pip install requests

# Install from requirements.txt
pip install -r requirements.txt

# Freeze current environment
pip freeze > requirements.txt

# Install in development mode
pip install -e .
```

## Best Practices

### 1. Module Design
- Keep modules focused and cohesive
- Use descriptive names
- Include docstrings
- Follow PEP 8 style guidelines

### 2. Package Organization
- Use flat structure when possible
- Group related functionality
- Separate tests from source code
- Include proper `__init__.py` files

### 3. Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import requests
import numpy as np

# Local imports
from .utils import helper_function
from ..models import User
```

### 4. Error Handling in Imports
```python
try:
    import optional_package
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    optional_package = None
```

## Common Patterns

### 1. Configuration Module
```python
# config.py
import os
from pathlib import Path

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### 2. Utility Module
```python
# utils.py
import logging
from functools import wraps
from typing import Any, Callable

def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def log_function_calls(func: Callable) -> Callable:
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper
```

### 3. Plugin System
```python
# plugin_base.py
from abc import ABC, abstractmethod

class Plugin(ABC):
    @abstractmethod
    def execute(self, data: Any) -> Any:
        pass

# plugin_manager.py
import importlib
import os
from pathlib import Path
from typing import Dict, List, Type

class PluginManager:
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Type[Plugin]] = {}
    
    def load_plugins(self) -> None:
        """Load all plugins from the plugin directory."""
        for file_path in self.plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            module_name = file_path.stem
            module = importlib.import_module(f"plugins.{module_name}")
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Plugin) and 
                    attr != Plugin):
                    self.plugins[module_name] = attr
    
    def get_plugin(self, name: str) -> Type[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)
```

## Testing Modules and Packages

### 1. Unit Testing
```python
# tests/test_math_utils.py
import unittest
from math_utils import add, multiply

class TestMathUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
    
    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)
        self.assertEqual(multiply(-1, 5), -5)

if __name__ == '__main__':
    unittest.main()
```

### 2. Package Testing
```python
# tests/test_package.py
import unittest
from my_package import function1, PublicClass

class TestPackage(unittest.TestCase):
    def test_function1(self):
        result = function1()
        self.assertIsNotNone(result)
    
    def test_public_class(self):
        obj = PublicClass()
        self.assertIsInstance(obj, PublicClass)
```

## Common Pitfalls

1. **Circular Imports** - Avoid importing modules that import each other
2. **Import Side Effects** - Avoid executing code at module level
3. **Missing `__init__.py`** - Required for packages
4. **Incorrect Import Paths** - Use relative imports within packages
5. **Namespace Pollution** - Use `__all__` to control exports

## Quick Checks

### Check 1: Module Import
```python
# What will this print?
import sys
print('math' in sys.modules)
import math
print('math' in sys.modules)
```

### Check 2: Package Structure
```python
# What will this print?
from my_package import *
# Assuming __all__ = ['function1']
print(function1())
print(_private_function())  # This will fail
```

### Check 3: Module Reloading
```python
# What happens here?
import my_module
print(my_module.version)  # '1.0'
# Modify my_module.py to change version to '2.0'
import importlib
importlib.reload(my_module)
print(my_module.version)  # What will this print?
```

## Lab Problems

### Lab 1: Create a Math Package
Build a comprehensive math package with multiple modules for different mathematical operations.

### Lab 2: Plugin System
Implement a plugin system that can dynamically load and execute plugins.

### Lab 3: Configuration Management
Create a configuration management system that loads settings from multiple sources.

### Lab 4: Package Distribution
Package your code for distribution using setuptools and publish to PyPI.

## AI Code Comparison
When working with AI-generated code, evaluate:
- **Import organization** - are imports properly organized and structured?
- **Package structure** - is the package hierarchy logical and maintainable?
- **Error handling** - are import errors handled gracefully?
- **Documentation** - are modules and packages properly documented?
- **Testing** - are there appropriate tests for the modules?

## Next Steps
- Learn about Python's packaging ecosystem
- Master virtual environment management
- Explore advanced import patterns
- Study package distribution and publishing
- Understand Python's module system internals
