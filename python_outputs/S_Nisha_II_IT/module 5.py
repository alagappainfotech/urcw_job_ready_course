   MODULE-5
Testing Module 5 Exercises...

1. Testing Math Package Creation:
Math package created successfully!
Package created at: math_package

2. Testing Plugin System:
Registered plugin: text_processor
Registered plugin: number_processor
Text processing result: HELLO WORLD
Number processing result: 15
Available plugins: ['text_processor', 'number_processor']

3. Testing Configuration Management:
Loaded configuration from dict
Loaded configuration from environment variables with prefix 'APP_'
Database URL: sqlite:///app.db
Debug mode: True
Secret key: secret123
Configuration sources: ['dict', 'environment']

4. Testing Module Loading:
Loaded module: math
Loaded module: os
Loaded modules: ['math', 'os']
Math module: <module 'math' from '/data/user/0/com.kvassyu.coding.py/files/PYROOT3/lib/python3.13/lib-dynload/math.cpython-313-aarch64-linux-android.so'>
OS module: <module 'os' (frozen)>

5. Testing Package Testing:
Test results:
  imports: 1 issues
    - Import error: No module named 'math_package'
  package_structure: 0 issues
  modules: 0 issues
6. Testing Virtual Environment Manager:

Creating virtual environment: venvs/test_env
Created virtual environment: venvs/test_env
Available virtual environments: ['test_env']

7. Testing Dependency Tracking:
Error analyzing module math: 'utf-8' codec can't decode byte 0xb7 in position 18: invalid start byte
Math module dependencies: []
OS module dependencies: ['abc', 'sys', 'stat', '_collections_abc', 'posix', 'posix', 'posixpath', 'posix', 'posix', 'nt', 'nt', 'ntpath', 'nt', 'nt', 'os.path', 'os', 'os.path', 'os', 'warnings', '_collections_abc', 'subprocess', 'io', 'nt']
All tracked dependencies: {'os': ['abc', 'sys', 'stat', '_collections_abc', 'posix', 'posix', 'posixpath', 'posix', 'posix', 'nt', 'nt', 'ntpath', 'nt', 'nt', 'os.path', 'os', 'os.path', 'os', 'warnings', '_collections_abc', 'subprocess', 'io', 'nt']}

All tests completed!