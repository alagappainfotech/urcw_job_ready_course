"""
Module 5: Modules and Packages - Exercises
Complete these exercises to master Python's module and package system.
"""

import os
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Exercise 1: Create a Math Package
def create_math_package():
    """
    Create a comprehensive math package with multiple modules.
    
    Structure:
    math_package/
    ├── __init__.py
    ├── basic_ops.py
    ├── advanced_ops.py
    ├── statistics.py
    └── geometry.py
    """
    
    # Create package directory
    package_dir = Path("math_package")
    package_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_content = '''
"""
Math Package - A comprehensive mathematics package.
"""

from .basic_ops import add, subtract, multiply, divide
from .advanced_ops import power, sqrt, factorial
from .statistics import mean, median, mode, standard_deviation
from .geometry import circle_area, rectangle_area, triangle_area

__version__ = "1.0.0"
__author__ = "Python Course"

__all__ = [
    'add', 'subtract', 'multiply', 'divide',
    'power', 'sqrt', 'factorial',
    'mean', 'median', 'mode', 'standard_deviation',
    'circle_area', 'rectangle_area', 'triangle_area'
]
'''
    
    with open(package_dir / "__init__.py", "w") as f:
        f.write(init_content)
    
    # Create basic_ops.py
    basic_ops_content = '''
"""
Basic mathematical operations.
"""

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    with open(package_dir / "basic_ops.py", "w") as f:
        f.write(basic_ops_content)
    
    # Create advanced_ops.py
    advanced_ops_content = '''
"""
Advanced mathematical operations.
"""

import math

def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent

def sqrt(number: float) -> float:
    """Calculate square root."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n > 170:  # Python's limit
        raise ValueError("Number too large for factorial calculation")
    return math.factorial(n)
'''
    
    with open(package_dir / "advanced_ops.py", "w") as f:
        f.write(advanced_ops_content)
    
    # Create statistics.py
    statistics_content = '''
"""
Statistical functions.
"""

from typing import List

def mean(numbers: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(numbers) / len(numbers)

def median(numbers: List[float]) -> float:
    """Calculate median."""
    if not numbers:
        raise ValueError("Cannot calculate median of empty list")
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]

def mode(numbers: List[float]) -> float:
    """Calculate mode (most frequent value)."""
    if not numbers:
        raise ValueError("Cannot calculate mode of empty list")
    
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    
    max_freq = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_freq]
    
    if len(modes) == len(numbers):
        raise ValueError("No unique mode found")
    
    return modes[0]

def standard_deviation(numbers: List[float]) -> float:
    """Calculate standard deviation."""
    if len(numbers) < 2:
        raise ValueError("Need at least 2 numbers for standard deviation")
    
    mean_val = mean(numbers)
    variance = sum((x - mean_val) ** 2 for x in numbers) / (len(numbers) - 1)
    return variance ** 0.5
'''
    
    with open(package_dir / "statistics.py", "w") as f:
        f.write(statistics_content)
    
    # Create geometry.py
    geometry_content = '''
"""
Geometric calculations.
"""

import math

def circle_area(radius: float) -> float:
    """Calculate area of a circle."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2

def rectangle_area(length: float, width: float) -> float:
    """Calculate area of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative")
    return length * width

def triangle_area(base: float, height: float) -> float:
    """Calculate area of a triangle."""
    if base < 0 or height < 0:
        raise ValueError("Base and height cannot be negative")
    return 0.5 * base * height
'''
    
    with open(package_dir / "geometry.py", "w") as f:
        f.write(geometry_content)
    
    print("Math package created successfully!")
    return package_dir

# Exercise 2: Plugin System
class Plugin(ABC):
    """Base class for all plugins."""
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute the plugin with given data."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""
        pass

class TextProcessor(Plugin):
    """Plugin for text processing."""
    
    @property
    def name(self) -> str:
        return "text_processor"
    
    def execute(self, data: str) -> str:
        """Convert text to uppercase."""
        return data.upper()

class NumberProcessor(Plugin):
    """Plugin for number processing."""
    
    @property
    def name(self) -> str:
        return "number_processor"
    
    def execute(self, data: List[float]) -> float:
        """Calculate sum of numbers."""
        return sum(data)

class PluginManager:
    """Manages and executes plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        self.plugins[plugin.name] = plugin
        print(f"Registered plugin: {plugin.name}")
    
    def execute_plugin(self, plugin_name: str, data: Any) -> Any:
        """Execute a specific plugin."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        plugin = self.plugins[plugin_name]
        return plugin.execute(data)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self.plugins.keys())

# Exercise 3: Configuration Management
class ConfigManager:
    """Manages configuration from multiple sources."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.sources: List[str] = []
    
    def load_from_dict(self, config_dict: Dict[str, Any], source: str = "dict") -> None:
        """Load configuration from dictionary."""
        self.config.update(config_dict)
        self.sources.append(source)
        print(f"Loaded configuration from {source}")
    
    def load_from_env(self, prefix: str = "") -> None:
        """Load configuration from environment variables."""
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    env_config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    env_config[config_key] = int(value)
                elif value.replace('.', '').isdigit():
                    env_config[config_key] = float(value)
                else:
                    env_config[config_key] = value
        
        self.config.update(env_config)
        self.sources.append("environment")
        print(f"Loaded configuration from environment variables with prefix '{prefix}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def get_sources(self) -> List[str]:
        """Get list of configuration sources."""
        return self.sources.copy()

# Exercise 4: Dynamic Module Loading
class ModuleLoader:
    """Dynamically loads and manages modules."""
    
    def __init__(self):
        self.loaded_modules: Dict[str, Any] = {}
    
    def load_module(self, module_name: str) -> Any:
        """Load a module dynamically."""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        try:
            module = importlib.import_module(module_name)
            self.loaded_modules[module_name] = module
            print(f"Loaded module: {module_name}")
            return module
        except ImportError as e:
            print(f"Failed to load module {module_name}: {e}")
            return None
    
    def reload_module(self, module_name: str) -> Any:
        """Reload a previously loaded module."""
        if module_name not in self.loaded_modules:
            print(f"Module {module_name} not loaded")
            return None
        
        try:
            module = importlib.reload(self.loaded_modules[module_name])
            self.loaded_modules[module_name] = module
            print(f"Reloaded module: {module_name}")
            return module
        except Exception as e:
            print(f"Failed to reload module {module_name}: {e}")
            return None
    
    def get_module(self, module_name: str) -> Any:
        """Get a loaded module."""
        return self.loaded_modules.get(module_name)
    
    def list_loaded_modules(self) -> List[str]:
        """List all loaded modules."""
        return list(self.loaded_modules.keys())

# Exercise 5: Package Testing Framework
class PackageTester:
    """Tests packages and modules for common issues."""
    
    def __init__(self):
        self.results: Dict[str, List[str]] = {}
    
    def test_imports(self, package_name: str) -> List[str]:
        """Test if package can be imported."""
        issues = []
        
        try:
            importlib.import_module(package_name)
        except ImportError as e:
            issues.append(f"Import error: {e}")
        except Exception as e:
            issues.append(f"Unexpected error: {e}")
        
        return issues
    
    def test_module_structure(self, module_path: str) -> List[str]:
        """Test module structure and organization."""
        issues = []
        module_file = Path(module_path)
        
        if not module_file.exists():
            issues.append(f"Module file not found: {module_path}")
            return issues
        
        # Check for docstring
        with open(module_file, 'r') as f:
            content = f.read()
            if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                issues.append("Module missing docstring")
        
        # Check for proper imports
        if 'import *' in content:
            issues.append("Module uses 'import *' which is not recommended")
        
        return issues
    
    def test_package_structure(self, package_path: str) -> List[str]:
        """Test package structure."""
        issues = []
        package_dir = Path(package_path)
        
        if not package_dir.exists():
            issues.append(f"Package directory not found: {package_path}")
            return issues
        
        # Check for __init__.py
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            issues.append("Package missing __init__.py file")
        
        # Check for __all__ in __init__.py
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
                if '__all__' not in content:
                    issues.append("Package __init__.py missing __all__ attribute")
        
        return issues
    
    def run_all_tests(self, package_name: str, package_path: str) -> Dict[str, List[str]]:
        """Run all tests on a package."""
        self.results = {}
        
        # Test imports
        self.results['imports'] = self.test_imports(package_name)
        
        # Test package structure
        self.results['package_structure'] = self.test_package_structure(package_path)
        
        # Test individual modules
        package_dir = Path(package_path)
        if package_dir.exists():
            module_issues = []
            for py_file in package_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    issues = self.test_module_structure(str(py_file))
                    module_issues.extend(issues)
            self.results['modules'] = module_issues
        
        return self.results

# Exercise 6: Virtual Environment Manager
class VirtualEnvManager:
    """Manages virtual environments."""
    
    def __init__(self, base_dir: str = "venvs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_venv(self, name: str) -> Path:
        """Create a virtual environment."""
        venv_path = self.base_dir / name
        
        if venv_path.exists():
            print(f"Virtual environment '{name}' already exists")
            return venv_path
        
        # In a real implementation, you would use subprocess to run:
        # python -m venv venv_path
        print(f"Creating virtual environment: {venv_path}")
        venv_path.mkdir(exist_ok=True)
        
        # Create a simple requirements.txt
        requirements_file = venv_path / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write("# Virtual environment requirements\n")
        
        return venv_path
    
    def list_venvs(self) -> List[str]:
        """List all virtual environments."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def delete_venv(self, name: str) -> bool:
        """Delete a virtual environment."""
        venv_path = self.base_dir / name
        
        if not venv_path.exists():
            print(f"Virtual environment '{name}' not found")
            return False
        
        # In a real implementation, you would use shutil.rmtree
        print(f"Deleting virtual environment: {venv_path}")
        return True

# Exercise 7: Module Dependency Tracker
class DependencyTracker:
    """Tracks module dependencies."""
    
    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}
    
    def analyze_module(self, module_name: str) -> List[str]:
        """Analyze a module's dependencies."""
        try:
            module = importlib.import_module(module_name)
            source_file = getattr(module, '__file__', None)
            
            if not source_file:
                return []
            
            with open(source_file, 'r') as f:
                content = f.read()
            
            # Simple regex to find import statements
            import re
            imports = []
            
            # Find import statements
            import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
            from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('import '):
                    match = re.match(import_pattern, line)
                    if match:
                        imports.append(match.group(1))
                elif line.startswith('from '):
                    match = re.match(from_pattern, line)
                    if match:
                        imports.append(match.group(1))
            
            self.dependencies[module_name] = imports
            return imports
            
        except Exception as e:
            print(f"Error analyzing module {module_name}: {e}")
            return []
    
    def get_dependencies(self, module_name: str) -> List[str]:
        """Get dependencies for a module."""
        return self.dependencies.get(module_name, [])
    
    def get_all_dependencies(self) -> Dict[str, List[str]]:
        """Get all tracked dependencies."""
        return self.dependencies.copy()

# Test Functions
def test_exercises():
    """Test all exercises."""
    print("Testing Module 5 Exercises...")
    
    # Test 1: Math Package
    print("\n1. Testing Math Package Creation:")
    package_dir = create_math_package()
    print(f"Package created at: {package_dir}")
    
    # Test 2: Plugin System
    print("\n2. Testing Plugin System:")
    manager = PluginManager()
    
    # Register plugins
    text_plugin = TextProcessor()
    number_plugin = NumberProcessor()
    
    manager.register_plugin(text_plugin)
    manager.register_plugin(number_plugin)
    
    # Test plugin execution
    result1 = manager.execute_plugin("text_processor", "hello world")
    result2 = manager.execute_plugin("number_processor", [1, 2, 3, 4, 5])
    
    print(f"Text processing result: {result1}")
    print(f"Number processing result: {result2}")
    print(f"Available plugins: {manager.list_plugins()}")
    
    # Test 3: Configuration Management
    print("\n3. Testing Configuration Management:")
    config = ConfigManager()
    
    # Load from dictionary
    config.load_from_dict({
        "database_url": "sqlite:///app.db",
        "debug": True,
        "max_connections": 100
    })
    
    # Load from environment (simulate)
    os.environ["APP_SECRET_KEY"] = "secret123"
    os.environ["APP_DEBUG"] = "true"
    config.load_from_env("APP_")
    
    print(f"Database URL: {config.get('database_url')}")
    print(f"Debug mode: {config.get('debug')}")
    print(f"Secret key: {config.get('secret_key')}")
    print(f"Configuration sources: {config.get_sources()}")
    
    # Test 4: Module Loading
    print("\n4. Testing Module Loading:")
    loader = ModuleLoader()
    
    # Load standard library modules
    math_module = loader.load_module("math")
    os_module = loader.load_module("os")
    
    print(f"Loaded modules: {loader.list_loaded_modules()}")
    print(f"Math module: {math_module}")
    print(f"OS module: {os_module}")
    
    # Test 5: Package Testing
    print("\n5. Testing Package Testing:")
    tester = PackageTester()
    
    # Test the math package we created
    results = tester.run_all_tests("math_package", str(package_dir))
    
    print("Test results:")
    for test_type, issues in results.items():
        print(f"  {test_type}: {len(issues)} issues")
        for issue in issues:
            print(f"    - {issue}")
    
    # Test 6: Virtual Environment Manager
    print("\n6. Testing Virtual Environment Manager:")
    venv_manager = VirtualEnvManager()
    
    # Create a test virtual environment
    venv_path = venv_manager.create_venv("test_env")
    print(f"Created virtual environment: {venv_path}")
    
    # List virtual environments
    venvs = venv_manager.list_venvs()
    print(f"Available virtual environments: {venvs}")
    
    # Test 7: Dependency Tracking
    print("\n7. Testing Dependency Tracking:")
    tracker = DependencyTracker()
    
    # Analyze some modules
    math_deps = tracker.analyze_module("math")
    os_deps = tracker.analyze_module("os")
    
    print(f"Math module dependencies: {math_deps}")
    print(f"OS module dependencies: {os_deps}")
    
    all_deps = tracker.get_all_dependencies()
    print(f"All tracked dependencies: {all_deps}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_exercises()
