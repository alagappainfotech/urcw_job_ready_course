# Module 14: Testing and Quality Assurance - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master unit testing, integration testing, and end-to-end testing
- Implement test-driven development (TDD) and behavior-driven development (BDD)
- Use testing frameworks like pytest, unittest, and nose
- Write effective test cases and test suites
- Implement continuous integration and automated testing
- Master code quality tools and static analysis
- Build robust, maintainable, and well-tested applications

## Core Concepts

### 1. Testing Fundamentals
Testing is the process of verifying that software behaves as expected under various conditions.

```python
# Basic testing concepts
def add(a, b):
    """Add two numbers"""
    return a + b

def test_add():
    """Test the add function"""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(1.5, 2.5) == 4.0

# Test categories
"""
Unit Tests: Test individual functions or methods
Integration Tests: Test interaction between components
System Tests: Test the entire system
Acceptance Tests: Test user requirements
Performance Tests: Test system performance
Security Tests: Test security vulnerabilities
"""
```

### 2. unittest Framework
Python's built-in testing framework provides a solid foundation for testing.

```python
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        from calculator import Calculator
        self.calc = Calculator()
    
    def tearDown(self):
        """Clean up after each test method"""
        self.calc = None
    
    def test_add(self):
        """Test addition operation"""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)
    
    def test_add_negative_numbers(self):
        """Test addition with negative numbers"""
        result = self.calc.add(-1, -2)
        self.assertEqual(result, -3)
    
    def test_add_float_numbers(self):
        """Test addition with float numbers"""
        result = self.calc.add(1.5, 2.5)
        self.assertAlmostEqual(result, 4.0, places=2)
    
    def test_divide(self):
        """Test division operation"""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5.0)
    
    def test_divide_by_zero(self):
        """Test division by zero raises exception"""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def test_divide_by_zero_message(self):
        """Test division by zero error message"""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        self.assertIn("Cannot divide by zero", str(context.exception))
    
    @unittest.skip("Skipping this test")
    def test_skip_example(self):
        """Example of a skipped test"""
        self.fail("This test should be skipped")
    
    @unittest.skipIf(sys.platform == "win32", "Not supported on Windows")
    def test_platform_specific(self):
        """Test that only runs on non-Windows platforms"""
        self.assertTrue(True)

class TestDatabase(unittest.TestCase):
    """Test cases for database operations"""
    
    def setUp(self):
        """Set up database connection for testing"""
        from database import Database
        self.db = Database(":memory:")  # Use in-memory database for testing
        self.db.create_tables()
    
    def tearDown(self):
        """Clean up database after each test"""
        self.db.close()
    
    def test_create_user(self):
        """Test user creation"""
        user_id = self.db.create_user("john", "john@example.com")
        self.assertIsNotNone(user_id)
        self.assertIsInstance(user_id, int)
    
    def test_get_user(self):
        """Test user retrieval"""
        user_id = self.db.create_user("jane", "jane@example.com")
        user = self.db.get_user(user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user["name"], "jane")
        self.assertEqual(user["email"], "jane@example.com")
    
    def test_user_not_found(self):
        """Test user not found scenario"""
        user = self.db.get_user(999)
        self.assertIsNone(user)

class TestMocking(unittest.TestCase):
    """Test cases demonstrating mocking"""
    
    def test_mock_external_api(self):
        """Test mocking external API calls"""
        with patch('requests.get') as mock_get:
            # Configure mock response
            mock_response = Mock()
            mock_response.json.return_value = {"status": "success", "data": "test"}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Test the function that uses the API
            from api_client import fetch_data
            result = fetch_data("https://api.example.com/data")
            
            # Verify the result
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["data"], "test")
            
            # Verify the API was called correctly
            mock_get.assert_called_once_with("https://api.example.com/data")
    
    def test_mock_database_connection(self):
        """Test mocking database connection"""
        with patch('database.Database') as mock_db_class:
            # Configure mock database
            mock_db = Mock()
            mock_db.get_user.return_value = {"id": 1, "name": "test"}
            mock_db_class.return_value = mock_db
            
            # Test the function that uses the database
            from user_service import get_user_info
            result = get_user_info(1)
            
            # Verify the result
            self.assertEqual(result["name"], "test")
            
            # Verify the database was called
            mock_db.get_user.assert_called_once_with(1)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
```

### 3. pytest Framework
pytest is a more modern and flexible testing framework.

```python
import pytest
from pytest import fixture, mark, raises
import tempfile
import os

# Fixtures for test setup
@fixture
def calculator():
    """Fixture providing a Calculator instance"""
    from calculator import Calculator
    return Calculator()

@fixture
def database():
    """Fixture providing a database connection"""
    from database import Database
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()

@fixture
def temp_file():
    """Fixture providing a temporary file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test data")
        f.flush()
        yield f.name
    os.unlink(f.name)

# Test functions using fixtures
def test_add_basic(calculator):
    """Test basic addition"""
    assert calculator.add(2, 3) == 5

def test_add_edge_cases(calculator):
    """Test addition edge cases"""
    assert calculator.add(0, 0) == 0
    assert calculator.add(-1, 1) == 0
    assert calculator.add(1.5, 2.5) == 4.0

def test_divide_by_zero(calculator):
    """Test division by zero"""
    with raises(ValueError, match="Cannot divide by zero"):
        calculator.divide(10, 0)

def test_database_operations(database):
    """Test database operations"""
    user_id = database.create_user("test", "test@example.com")
    assert user_id is not None
    
    user = database.get_user(user_id)
    assert user["name"] == "test"
    assert user["email"] == "test@example.com"

def test_file_operations(temp_file):
    """Test file operations"""
    assert os.path.exists(temp_file)
    
    with open(temp_file, 'r') as f:
        content = f.read()
    assert content == "test data"

# Parametrized tests
@mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (1.5, 2.5, 4.0)
])
def test_add_parametrized(calculator, a, b, expected):
    """Test addition with multiple parameters"""
    assert calculator.add(a, b) == expected

@mark.parametrize("input_value,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
    ("123", "123")
])
def test_upper_case(input_value, expected):
    """Test string upper case conversion"""
    assert input_value.upper() == expected

# Test classes
class TestCalculatorClass:
    """Test class using pytest"""
    
    def test_add(self, calculator):
        """Test addition"""
        assert calculator.add(2, 3) == 5
    
    def test_subtract(self, calculator):
        """Test subtraction"""
        assert calculator.subtract(5, 3) == 2
    
    def test_multiply(self, calculator):
        """Test multiplication"""
        assert calculator.multiply(4, 5) == 20
    
    def test_divide(self, calculator):
        """Test division"""
        assert calculator.divide(10, 2) == 5.0

# Markers for test categorization
@mark.slow
def test_slow_operation():
    """Test that takes a long time"""
    import time
    time.sleep(2)
    assert True

@mark.integration
def test_integration():
    """Integration test"""
    assert True

@mark.unit
def test_unit():
    """Unit test"""
    assert True

# Configuration in pytest.ini
"""
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
"""
```

### 4. Test-Driven Development (TDD)
TDD is a development approach where tests are written before the code.

```python
# TDD Example: Building a Stack class

# Step 1: Write a failing test
def test_stack_creation():
    """Test that a stack can be created"""
    from stack import Stack
    stack = Stack()
    assert stack.is_empty()
    assert stack.size() == 0

# Step 2: Write minimal code to make the test pass
class Stack:
    def __init__(self):
        self._items = []
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)

# Step 3: Write more tests
def test_stack_push():
    """Test pushing items to stack"""
    stack = Stack()
    stack.push(1)
    assert not stack.is_empty()
    assert stack.size() == 1

def test_stack_pop():
    """Test popping items from stack"""
    stack = Stack()
    stack.push(1)
    stack.push(2)
    assert stack.pop() == 2
    assert stack.pop() == 1
    assert stack.is_empty()

def test_stack_peek():
    """Test peeking at top item"""
    stack = Stack()
    stack.push(1)
    stack.push(2)
    assert stack.peek() == 2
    assert stack.size() == 2

# Step 4: Implement the methods
class Stack:
    def __init__(self):
        self._items = []
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self._items[-1]

# Step 5: Refactor and improve
class Stack:
    def __init__(self):
        self._items = []
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self._items[-1]
    
    def __str__(self):
        return f"Stack({self._items})"
    
    def __repr__(self):
        return f"Stack({self._items})"
```

### 5. Behavior-Driven Development (BDD)
BDD uses natural language to describe behavior and tests.

```python
# Using pytest-bdd for BDD
from pytest_bdd import given, when, then, scenario
import pytest

@scenario('features/calculator.feature', 'Add two numbers')
def test_add_two_numbers():
    pass

@given('I have a calculator')
def calculator():
    from calculator import Calculator
    return Calculator()

@when('I add 2 and 3')
def add_numbers(calculator):
    calculator.add(2, 3)

@then('the result should be 5')
def check_result(calculator):
    assert calculator.get_result() == 5

# Feature file: features/calculator.feature
"""
Feature: Calculator
  As a user
  I want to use a calculator
  So that I can perform mathematical operations

  Scenario: Add two numbers
    Given I have a calculator
    When I add 2 and 3
    Then the result should be 5

  Scenario: Divide by zero
    Given I have a calculator
    When I divide 10 by 0
    Then I should get an error
"""
```

### 6. Integration Testing
Integration tests verify that different components work together correctly.

```python
import pytest
import requests
from unittest.mock import patch
import json

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test application"""
        from app import create_app
        app = create_app(testing=True)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_user_registration(self, client):
        """Test user registration endpoint"""
        response = client.post('/api/users', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['username'] == 'testuser'
        assert data['email'] == 'test@example.com'
        assert 'id' in data
    
    def test_user_login(self, client):
        """Test user login endpoint"""
        # First register a user
        client.post('/api/users', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        
        # Then login
        response = client.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'testpass123'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'token' in data
    
    def test_protected_endpoint(self, client):
        """Test protected endpoint with authentication"""
        # Register and login
        client.post('/api/users', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        
        login_response = client.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'testpass123'
        })
        
        token = json.loads(login_response.data)['token']
        
        # Access protected endpoint
        response = client.get('/api/profile', headers={
            'Authorization': f'Bearer {token}'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['username'] == 'testuser'

class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @pytest.fixture
    def db(self):
        """Create test database"""
        from database import Database
        db = Database(":memory:")
        db.create_tables()
        yield db
        db.close()
    
    def test_user_crud_operations(self, db):
        """Test complete CRUD operations for users"""
        # Create
        user_id = db.create_user("testuser", "test@example.com")
        assert user_id is not None
        
        # Read
        user = db.get_user(user_id)
        assert user['username'] == "testuser"
        assert user['email'] == "test@example.com"
        
        # Update
        db.update_user(user_id, username="updateduser")
        updated_user = db.get_user(user_id)
        assert updated_user['username'] == "updateduser"
        
        # Delete
        db.delete_user(user_id)
        deleted_user = db.get_user(user_id)
        assert deleted_user is None
    
    def test_user_posts_relationship(self, db):
        """Test user-posts relationship"""
        # Create user
        user_id = db.create_user("testuser", "test@example.com")
        
        # Create posts
        post1_id = db.create_post(user_id, "First Post", "Content 1")
        post2_id = db.create_post(user_id, "Second Post", "Content 2")
        
        # Get user posts
        posts = db.get_user_posts(user_id)
        assert len(posts) == 2
        assert posts[0]['title'] == "First Post"
        assert posts[1]['title'] == "Second Post"
```

### 7. Performance Testing
Performance tests ensure the application meets performance requirements.

```python
import pytest
import time
import psutil
import os
from memory_profiler import profile

class TestPerformance:
    """Performance tests"""
    
    def test_response_time(self):
        """Test API response time"""
        import requests
        
        start_time = time.time()
        response = requests.get('http://localhost:5000/api/health')
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200
    
    def test_memory_usage(self):
        """Test memory usage during operation"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operation
        data = [i for i in range(1000000)]
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
        # Clean up
        del data
    
    @profile
    def test_memory_profiling(self):
        """Test with memory profiling"""
        # This function will be profiled for memory usage
        data = []
        for i in range(10000):
            data.append(i * 2)
        return data
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import threading
        import requests
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = requests.get('http://localhost:5000/api/data')
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create 10 concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_database_performance(self):
        """Test database performance"""
        from database import Database
        
        db = Database(":memory:")
        db.create_tables()
        
        # Test bulk insert performance
        start_time = time.time()
        
        for i in range(1000):
            db.create_user(f"user{i}", f"user{i}@example.com")
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Should insert 1000 users in less than 5 seconds
        assert insert_time < 5.0
        
        # Test query performance
        start_time = time.time()
        
        users = db.get_all_users()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Should query 1000 users in less than 1 second
        assert query_time < 1.0
        assert len(users) == 1000
        
        db.close()
```

### 8. Code Quality Tools
Tools for maintaining code quality and catching issues early.

```python
# flake8 configuration
"""
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,env
"""

# black configuration
"""
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
"""

# mypy configuration
"""
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
"""

# pytest configuration
"""
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""

# pre-commit configuration
"""
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
"""
```

## Best Practices

### 1. Test Organization
```python
# tests/
# ├── __init__.py
# ├── conftest.py
# ├── unit/
# │   ├── __init__.py
# │   ├── test_calculator.py
# │   └── test_database.py
# ├── integration/
# │   ├── __init__.py
# │   ├── test_api.py
# │   └── test_database_integration.py
# └── fixtures/
#     ├── __init__.py
#     └── sample_data.py

# conftest.py - Shared fixtures
import pytest
from app import create_app
from database import Database

@pytest.fixture(scope="session")
def app():
    """Create application for testing"""
    app = create_app(testing=True)
    return app

@pytest.fixture(scope="session")
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture(scope="function")
def db():
    """Create test database"""
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()

@pytest.fixture(scope="function")
def sample_user(db):
    """Create sample user for testing"""
    user_id = db.create_user("testuser", "test@example.com")
    return db.get_user(user_id)
```

### 2. Test Data Management
```python
# fixtures/sample_data.py
import json
import os

class SampleData:
    """Manage sample data for testing"""
    
    @staticmethod
    def load_json(filename):
        """Load JSON data from file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_sample_users():
        """Get sample user data"""
        return [
            {"username": "user1", "email": "user1@example.com"},
            {"username": "user2", "email": "user2@example.com"},
            {"username": "user3", "email": "user3@example.com"}
        ]
    
    @staticmethod
    def get_sample_posts():
        """Get sample post data"""
        return [
            {"title": "Post 1", "content": "Content 1"},
            {"title": "Post 2", "content": "Content 2"},
            {"title": "Post 3", "content": "Content 3"}
        ]

# Usage in tests
def test_with_sample_data(db, sample_user):
    """Test using sample data"""
    posts = SampleData.get_sample_posts()
    for post in posts:
        db.create_post(sample_user['id'], post['title'], post['content'])
    
    user_posts = db.get_user_posts(sample_user['id'])
    assert len(user_posts) == 3
```

### 3. Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 .
        black --check .
        isort --check-only .
    
    - name: Run type checking
      run: mypy .
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Quick Checks

### Check 1: Test Structure
```python
# What will this test do?
def test_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5
    assert calc.divide(10, 0) == 0  # This will fail
```

### Check 2: Mocking
```python
# What will this test verify?
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"status": "ok"}
    result = fetch_data("https://api.example.com")
    assert result["status"] == "ok"
    mock_get.assert_called_once_with("https://api.example.com")
```

### Check 3: Fixtures
```python
# What will this fixture provide?
@pytest.fixture
def database():
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()
```

## Lab Problems

### Lab 1: Test Suite for E-commerce
Build a comprehensive test suite for an e-commerce application including unit, integration, and end-to-end tests.

### Lab 2: Performance Testing Framework
Create a performance testing framework that can measure response times, memory usage, and throughput.

### Lab 3: Test Automation Pipeline
Set up a complete CI/CD pipeline with automated testing, code quality checks, and deployment.

### Lab 4: Test Data Management
Implement a robust test data management system that can generate, clean, and manage test data across different environments.

## AI Code Comparison
When working with AI-generated test code, evaluate:
- **Test coverage** - are all critical paths and edge cases covered?
- **Test quality** - are tests meaningful and not just testing implementation details?
- **Test maintainability** - are tests easy to understand and maintain?
- **Test performance** - are tests fast and efficient?
- **Test reliability** - are tests stable and not flaky?

## Next Steps
- Learn about advanced testing techniques like property-based testing
- Master test automation and continuous testing
- Explore testing in microservices and distributed systems
- Study performance testing and load testing
- Understand testing in cloud and containerized environments
