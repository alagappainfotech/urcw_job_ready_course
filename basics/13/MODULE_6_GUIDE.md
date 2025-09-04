# Module 6: Packages & Ecosystem

## Learning Objectives
By the end of this module, you will be able to:
- Structure large Python projects with packages and modules
- Leverage the Python standard library effectively
- Integrate third-party packages from PyPI
- Manage dependencies and virtual environments
- Understand modern Python development practices
- Build and distribute Python packages

## Core Concepts

### 6.1 Python Package System
Understanding Python's package and module system:
- **Modules:** Single Python files containing code
- **Packages:** Directories containing multiple modules
- **Import System:** How Python finds and loads modules
- **Namespace Packages:** Packages without __init__.py files
- **Package Discovery:** How Python locates packages

### 6.2 Standard Library Mastery
Leveraging Python's "batteries included" philosophy:
- **Core Modules:** Essential built-in modules
- **Data Processing:** csv, json, xml, sqlite3
- **System Integration:** os, sys, subprocess, pathlib
- **Networking:** urllib, http, socket
- **Development Tools:** unittest, logging, argparse

### 6.3 Third-Party Package Management
Working with the Python Package Index (PyPI):
- **Package Installation:** pip and alternative tools
- **Dependency Management:** requirements.txt and virtual environments
- **Package Discovery:** Finding and evaluating packages
- **Version Management:** Handling package versions and conflicts
- **Security:** Managing package security and vulnerabilities

### 6.4 Modern Development Practices
Contemporary Python development workflows:
- **Virtual Environments:** venv, virtualenv, conda
- **Dependency Management:** pip-tools, poetry, pipenv
- **Package Building:** setuptools, wheel, build
- **Testing:** pytest, unittest, coverage
- **Code Quality:** black, flake8, mypy, isort

## Key Topics

### 6.5 Package Structure and Organization
- **Package Layout:** Organizing code into logical packages
- **__init__.py Files:** Package initialization and public API
- **Relative vs. Absolute Imports:** Import best practices
- **Package Discovery:** Making packages discoverable
- **Namespace Packages:** Modern package organization
 - **__all__ and private names:** Control public API and hide internals with `_name`

### 6.6 Standard Library Deep Dive
- **Data Types:** collections, itertools, functools
- **File Operations:** pathlib, shutil, tempfile
- **Text Processing:** string, re, difflib
- **Date and Time:** datetime, time, calendar
- **Mathematical:** math, statistics, random
- **Concurrency:** threading, multiprocessing, asyncio

### 6.7 Package Management Tools
- **pip:** Basic package installation and management
- **virtualenv/venv:** Environment isolation
- **pip-tools:** Dependency compilation and management
- **poetry:** Modern dependency and project management
- **conda:** Cross-platform package management
- **pipenv:** Pipfile-based dependency management

### 6.8 Package Development and Distribution
- **Package Creation:** Building distributable packages
- **setup.py and pyproject.toml:** Package configuration
- **Wheel Distribution:** Modern package distribution format
- **PyPI Publishing:** Sharing packages with the community
- **Package Testing:** Testing packages before distribution

### 6.9 Development Environment Setup
- **IDE Configuration:** Setting up development environments
- **Linting and Formatting:** Code quality tools
- **Testing Frameworks:** Unit testing and test automation
- **Documentation:** Generating and maintaining documentation
- **CI/CD:** Continuous integration and deployment

## Hands-on Learning Path

### Quick Checks (Immediate Reinforcement)
1. **Package Structure:** Design appropriate package organization
2. **Import Analysis:** Understand import resolution and best practices
3. **Dependency Management:** Choose appropriate package management tools
4. **Standard Library:** Identify relevant standard library modules

### Try This Exercises (Hands-on Application)
1. **Package Creation:** Build a complete Python package
2. **Dependency Management:** Set up virtual environments and dependencies
3. **Standard Library Usage:** Implement solutions using built-in modules
4. **Package Distribution:** Prepare packages for distribution

### Lab Problems (Critical Thinking)
1. **Web Scraper Package:** Build a reusable web scraping package
2. **Data Processing Library:** Create a data manipulation package
3. **Configuration Manager:** Develop a flexible configuration package
4. **API Client Library:** Build a third-party API client package

## Assessment Criteria
- **Package Design:** Well-structured, maintainable package organization
- **Dependency Management:** Proper use of virtual environments and dependency tools
- **Standard Library Usage:** Effective use of built-in modules
- **Code Quality:** Following modern Python development practices
- **Documentation:** Clear package documentation and usage examples
- **Testing:** Comprehensive testing of package functionality

## Resources
### Additional Walkthrough
- TestPyPI publishing (read-only walkthrough): build wheel with `python -m build`; upload to TestPyPI with `twine upload --repository testpypi dist/*`; install via `pip install --index-url https://test.pypi.org/simple/ your-package`.

### AI-comparison callout
For package creation labs, compare:
- Correct directory structure and imports
- Minimal, clear public API via `__all__`
- Dependency pinning and README completeness
- [Python Package User Guide](https://packaging.python.org/)
- [Python Standard Library](https://docs.python.org/3/library/)
- [PyPI - Python Package Index](https://pypi.org/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)
- [Python Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)

## Next Steps
After completing Module 6, you'll be ready to explore data handling, persistence, and exploration in Module 7. The package management skills you develop here will be essential for building and maintaining large-scale Python applications.
