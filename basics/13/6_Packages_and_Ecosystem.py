####################################################
## 6. Packages & Ecosystem
## Module 6: Packages & Ecosystem
####################################################

print("Module 6: Packages & Ecosystem")
print("=" * 60)

# =============================================================================
# PACKAGES AND MODULES OVERVIEW
# =============================================================================

# Modules are single .py files; packages are directories with __init__.py (or
# namespace packages without it) that contain modules and subpackages.

import importlib
import sys
import pkgutil

print("\nPython Module Search Path (sys.path):")
for p in sys.path[:5]:
    print(" ", p)

print("\nDiscovering installed top-level packages (first 20):")
found = [m.name for m in pkgutil.iter_modules()]
for name in sorted(found)[:20]:
    print(" ", name)

# =============================================================================
# STANDARD LIBRARY SAMPLER
# =============================================================================

print("\nStandard library sampler:")
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import csv
import statistics

# pathlib demonstration
project_root = Path.cwd()
print(f"Current working directory: {project_root}")
print(f"Entries in cwd (first 5): {[p.name for p in list(project_root.iterdir())[:5]]}")

# collections demonstration
counts = Counter(["a", "b", "a", "c", "b", "a"])
print("Counter example:", counts)

by_category = defaultdict(list)
by_category["fruits"].extend(["apple", "banana"]) 
by_category["veggies"].append("carrot")
print("defaultdict example:", dict(by_category))

# datetime demonstration
now = datetime.now()
print("Now:", now.isoformat(), " | +3 days:", (now + timedelta(days=3)).isoformat())

# json demonstration
obj = {"name": "Alice", "age": 30}
print("JSON dumps:", json.dumps(obj))

# csv demonstration (to string)
import io
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow(["name", "score"]) 
writer.writerows([["Alice", 95], ["Bob", 88]])
print("CSV content:\n" + buf.getvalue())

# statistics demonstration
print("Mean of [1,2,3,4,5]:", statistics.mean([1,2,3,4,5]))

# =============================================================================
# VIRTUAL ENVIRONMENTS & DEPENDENCIES (CONCEPTUAL)
# =============================================================================

print("\nDependency management tips:")
print("- Use venv for per-project environments")
print("- Pin versions in requirements.txt for reproducibility")
print("- Consider tools: pip-tools, poetry, or pipenv for advanced workflows")

# =============================================================================
# IMPORT PATTERNS
# =============================================================================

print("\nImport patterns:")
print("- Absolute imports preferred: from mypkg.utils import helper")
print("- Relative imports within a package: from .utils import helper")
print("- Avoid wildcard imports in production code")

# Dynamic import example
module_name = "json"
json_module = importlib.import_module(module_name)
print("Dynamically imported:", json_module.__name__)

# =============================================================================
# SIMPLE PACKAGE LAYOUT EXAMPLE (DOC ONLY)
# =============================================================================

print("\nSuggested package layout:")
print("""
mypackage/
  __init__.py          # Define public API via __all__
  utils.py             # Helpers
  processing/
    __init__.py
    cleaning.py        # Clean/normalize data
    transform.py       # Transform data
  io/
    __init__.py
    readers.py         # CSV/JSON readers
    writers.py         # CSV/JSON writers
tests/
  test_utils.py
  test_processing.py
""")

print("\n" + "="*60)
print("MODULE 6 COMPLETE!")
print("Next: Module 7 - Data Handling, Persistence & Exploration")
print("="*60)


