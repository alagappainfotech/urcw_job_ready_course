####################################################
## 4. Python Programs & Filesystem Interaction
## Module 4: Python Programs & Filesystem Interaction
####################################################

# =============================================================================
# LEARNING OBJECTIVES:
# 1. Create standalone Python scripts and applications
# 2. Interact with the operating system and filesystem effectively
# 3. Manage command-line interfaces and argument parsing
# 4. Handle file operations and directory management
# 5. Implement cross-platform compatibility considerations
# 6. Build practical command-line utilities
# =============================================================================

# =============================================================================
# PROGRAM STRUCTURE PRINCIPLES:
# - Scripts vs. Modules: When to create standalone scripts vs. reusable modules
# - Entry Points: Using if __name__ == "__main__": effectively
# - Program Lifecycle: Initialization, execution, and cleanup phases
# - Error Handling: Graceful program termination and error reporting
# =============================================================================

print("Module 4: Python Programs & Filesystem Interaction")
print("=" * 60)

# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================

print("\n" + "="*60)
print("COMMAND-LINE ARGUMENT PARSING")
print("="*60)

import sys
import argparse
from pathlib import Path

# Basic sys.argv usage
print("1. Basic sys.argv usage:")
print(f"Script name: {sys.argv[0]}")
print(f"Number of arguments: {len(sys.argv) - 1}")
if len(sys.argv) > 1:
    print(f"Arguments: {sys.argv[1:]}")

# Advanced argparse usage
def create_argument_parser():
    """Create a comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="A sample Python program demonstrating CLI features",
        epilog="Example: python script.py --input file.txt --output result.txt --verbose"
    )
    
    # Positional arguments
    parser.add_argument("input_file", help="Input file to process")
    
    # Optional arguments
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--format", choices=["txt", "json", "csv"], default="txt", 
                       help="Output format (default: txt)")
    parser.add_argument("--count", type=int, default=1, 
                       help="Number of times to process (default: 1)")
    
    # Mutually exclusive group
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create", action="store_true", help="Create new file")
    group.add_argument("--update", action="store_true", help="Update existing file")
    
    return parser

# Example of using the parser (commented out for demo)
# parser = create_argument_parser()
# args = parser.parse_args()

print("2. Argument parser created with comprehensive options")

# =============================================================================
# FILESYSTEM OPERATIONS WITH PATHLIB
# =============================================================================

print("\n" + "="*60)
print("FILESYSTEM OPERATIONS WITH PATHLIB")
print("="*60)

# Modern path handling with pathlib
print("1. Pathlib operations:")

# Create Path objects
current_dir = Path.cwd()
home_dir = Path.home()
script_path = Path(__file__)

print(f"Current directory: {current_dir}")
print(f"Home directory: {home_dir}")
print(f"Script path: {script_path}")
print(f"Script name: {script_path.name}")
print(f"Script parent: {script_path.parent}")

# Path operations
test_path = Path("test_directory") / "subdirectory" / "file.txt"
print(f"Constructed path: {test_path}")
print(f"Path parts: {test_path.parts}")
print(f"Path suffix: {test_path.suffix}")
print(f"Path stem: {test_path.stem}")

# Path existence and properties
print(f"Script exists: {script_path.exists()}")
print(f"Script is file: {script_path.is_file()}")
print(f"Script is directory: {script_path.is_dir()}")

# =============================================================================
# FILE OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("FILE OPERATIONS")
print("="*60)

# File reading and writing
print("1. File operations:")

# Create a sample file
sample_content = """This is a sample file for demonstration.
It contains multiple lines of text.
We'll use this to demonstrate file operations.
"""

sample_file = Path("sample.txt")

# Write to file
try:
    sample_file.write_text(sample_content)
    print(f"✓ Created file: {sample_file}")
except Exception as e:
    print(f"✗ Error creating file: {e}")

# Read from file
try:
    content = sample_file.read_text()
    print(f"✓ Read file content ({len(content)} characters)")
    print(f"First line: {content.splitlines()[0]}")
except Exception as e:
    print(f"✗ Error reading file: {e}")

# Read file line by line
try:
    with sample_file.open('r') as f:
        lines = f.readlines()
    print(f"✓ Read {len(lines)} lines from file")
except Exception as e:
    print(f"✗ Error reading lines: {e}")

# =============================================================================
# DIRECTORY OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("DIRECTORY OPERATIONS")
print("="*60)

# Directory creation and management
print("1. Directory operations:")

# Create test directory
test_dir = Path("test_operations")
try:
    test_dir.mkdir(exist_ok=True)
    print(f"✓ Created directory: {test_dir}")
except Exception as e:
    print(f"✗ Error creating directory: {e}")

# Create subdirectories
subdirs = ["docs", "data", "scripts"]
for subdir in subdirs:
    subdir_path = test_dir / subdir
    try:
        subdir_path.mkdir(exist_ok=True)
        print(f"✓ Created subdirectory: {subdir_path}")
    except Exception as e:
        print(f"✗ Error creating subdirectory: {e}")

# List directory contents
try:
    print(f"\nContents of {test_dir}:")
    for item in test_dir.iterdir():
        item_type = "directory" if item.is_dir() else "file"
        print(f"  {item.name} ({item_type})")
except Exception as e:
    print(f"✗ Error listing directory: {e}")

# =============================================================================
# CROSS-PLATFORM COMPATIBILITY
# =============================================================================

print("\n" + "="*60)
print("CROSS-PLATFORM COMPATIBILITY")
print("="*60)

import os
import platform

print("1. Platform information:")
print(f"Operating system: {platform.system()}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"Python version: {platform.python_version()}")

# Path separators
print(f"\nPath separators:")
print(f"OS path separator: {os.sep}")
print(f"Path separator (pathlib): {Path('/')}")

# Environment variables
print(f"\nEnvironment variables:")
print(f"HOME: {os.environ.get('HOME', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')[:50]}...")

# =============================================================================
# INPUT/OUTPUT REDIRECTION
# =============================================================================

print("\n" + "="*60)
print("INPUT/OUTPUT REDIRECTION")
print("="*60)

# Standard streams
print("1. Standard streams:")
print(f"stdin: {sys.stdin}")
print(f"stdout: {sys.stdout}")
print(f"stderr: {sys.stderr}")

# Writing to different streams
print("\n2. Writing to different streams:")
print("This goes to stdout")
print("This goes to stderr", file=sys.stderr)

# =============================================================================
# PRACTICAL EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("PRACTICAL EXAMPLES")
print("="*60)

# Example 1: File organizer
def organize_files_by_extension(directory: Path):
    """Organize files by their extensions."""
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    # Create extension directories
    extensions = set()
    for file_path in directory.iterdir():
        if file_path.is_file():
            extensions.add(file_path.suffix or "no_extension")
    
    for ext in extensions:
        ext_dir = directory / f"_{ext[1:] if ext != 'no_extension' else 'no_extension'}"
        ext_dir.mkdir(exist_ok=True)
    
    # Move files
    moved_count = 0
    for file_path in directory.iterdir():
        if file_path.is_file():
            ext = file_path.suffix or "no_extension"
            ext_dir = directory / f"_{ext[1:] if ext != 'no_extension' else 'no_extension'}"
            new_path = ext_dir / file_path.name
            
            try:
                file_path.rename(new_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
    
    print(f"Organized {moved_count} files by extension")

print("1. File organizer example:")
# organize_files_by_extension(Path("test_operations"))

# Example 2: Log file analyzer
def analyze_log_file(log_path: Path):
    """Analyze a log file and provide statistics."""
    if not log_path.exists():
        print(f"Error: {log_path} does not exist")
        return
    
    try:
        with log_path.open('r') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        error_lines = [line for line in lines if 'ERROR' in line.upper()]
        warning_lines = [line for line in lines if 'WARNING' in line.upper()]
        
        print(f"Log file analysis for {log_path.name}:")
        print(f"  Total lines: {total_lines}")
        print(f"  Error lines: {len(error_lines)}")
        print(f"  Warning lines: {len(warning_lines)}")
        
        if error_lines:
            print(f"  First error: {error_lines[0].strip()}")
        
    except Exception as e:
        print(f"Error analyzing log file: {e}")

print("\n2. Log analyzer example:")
# analyze_log_file(Path("sample.txt"))

# Example 3: System information gatherer
def gather_system_info():
    """Gather and display system information."""
    info = {
        "Platform": platform.platform(),
        "System": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "Current Directory": Path.cwd(),
        "Home Directory": Path.home(),
    }
    
    print("System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

print("\n3. System information:")
gather_system_info()

# =============================================================================
# ERROR HANDLING FOR SYSTEM OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("ERROR HANDLING FOR SYSTEM OPERATIONS")
print("="*60)

def safe_file_operation(operation_func, *args, **kwargs):
    """Safely execute file operations with error handling."""
    try:
        return operation_func(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return None
    except OSError as e:
        print(f"OS error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
def read_file_safely(file_path: Path):
    """Safely read a file."""
    return file_path.read_text()

print("Safe file operations:")
result = safe_file_operation(read_file_safely, Path("nonexistent.txt"))
if result is None:
    print("File operation failed safely")

# =============================================================================
# BEST PRACTICES
# =============================================================================

print("\n" + "="*60)
print("BEST PRACTICES")
print("="*60)

print("""
1. PROGRAM STRUCTURE:
   - Use if __name__ == "__main__": for script entry points
   - Organize code into functions and classes
   - Use proper error handling and logging
   - Provide clear help and usage information

2. COMMAND-LINE INTERFACES:
   - Use argparse for complex argument parsing
   - Provide comprehensive help text
   - Validate input arguments
   - Use meaningful exit codes

3. FILE OPERATIONS:
   - Use pathlib for modern path handling
   - Always handle file operation errors
   - Use context managers (with statements)
   - Consider cross-platform compatibility

4. ERROR HANDLING:
   - Catch specific exceptions when possible
   - Provide meaningful error messages
   - Log errors appropriately
   - Clean up resources on errors

5. CROSS-PLATFORM COMPATIBILITY:
   - Use pathlib instead of os.path when possible
   - Be aware of different line ending conventions
   - Handle different file permission systems
   - Test on multiple platforms

6. RESOURCE MANAGEMENT:
   - Use context managers for file operations
   - Clean up temporary files
   - Handle large files efficiently
   - Monitor memory usage
""")

# =============================================================================
# CLEANUP
# =============================================================================

print("\n" + "="*60)
print("CLEANUP")
print("="*60)

# Clean up created files and directories
try:
    if sample_file.exists():
        sample_file.unlink()
        print(f"✓ Removed sample file: {sample_file}")
    
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"✓ Removed test directory: {test_dir}")
    
    print("✓ Cleanup completed successfully")
except Exception as e:
    print(f"✗ Error during cleanup: {e}")

print("\n" + "="*60)
print("MODULE 4 COMPLETE!")
print("Next: Module 5 - Object-Oriented Programming & Advanced Features")
print("="*60)
