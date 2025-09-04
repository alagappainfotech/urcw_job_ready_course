# Module 4: Python Programs & Filesystem Interaction

## Learning Objectives
By the end of this module, you will be able to:
- Create standalone Python scripts and applications
- Interact with the operating system and filesystem effectively
- Manage command-line interfaces and argument parsing
- Handle file operations and directory management
- Implement cross-platform compatibility considerations
- Build practical command-line utilities

## Core Concepts

### 4.1 Python Program Structure
Understanding how to structure Python programs for different use cases:
- **Scripts vs. Modules:** When to create standalone scripts vs. reusable modules
- **Entry Points:** Using `if __name__ == "__main__":` effectively
- **Program Lifecycle:** Initialization, execution, and cleanup phases
- **Error Handling:** Graceful program termination and error reporting

### 4.2 Command-Line Interface Design
Creating user-friendly command-line interfaces:
- **Argument Parsing:** Using `argparse` for robust CLI design
- **Help Systems:** Comprehensive help and usage information
- **Input Validation:** Validating command-line arguments
- **Interactive vs. Non-interactive:** Designing for different usage patterns

### 4.3 Filesystem Operations
Working with files and directories effectively:
- **Path Management:** Using `pathlib` for modern path handling
- **File Operations:** Reading, writing, and manipulating files
- **Directory Operations:** Creating, navigating, and managing directories
- **Cross-platform Compatibility:** Writing code that works on different operating systems

### 4.4 System Integration
Integrating Python programs with the operating system:
- **Environment Variables:** Accessing and setting system environment variables
- **Process Management:** Running external programs and managing processes
- **File Permissions:** Understanding and managing file system permissions
- **Resource Management:** Proper cleanup and resource handling

## Key Topics

### 4.5 Creating Standalone Programs
- **Script Structure:** Organizing code for standalone execution
- **Main Function Pattern:** Using main() functions for better organization
- **Exit Codes:** Proper program termination with meaningful exit codes
- **Logging:** Implementing logging for program execution tracking

### 4.6 Command-Line Arguments
- **sys.argv:** Basic command-line argument access
- **argparse Module:** Advanced argument parsing and validation
- **Subcommands:** Creating programs with multiple commands
- **Configuration Files:** Combining CLI arguments with configuration files

### 4.7 File and Directory Operations
- **pathlib Module:** Modern path handling and manipulation
- **os and os.path:** Traditional file system operations
- **shutil Module:** High-level file operations
- **glob Module:** Pattern-based file searching
- **tempfile Module:** Working with temporary files and directories

### 4.8 Input/Output Redirection
- **Standard Streams:** stdin, stdout, stderr manipulation
- **File Redirection:** Reading from and writing to files
- **Piping:** Connecting programs through standard streams
- **Buffering:** Understanding and controlling I/O buffering

### 4.9 Cross-Platform Considerations
- **Path Separators:** Handling different path formats
- **Line Endings:** Managing different line ending conventions
- **File Permissions:** Understanding permission systems
- **Environment Differences:** Handling OS-specific behaviors

## Hands-on Learning Path

### Quick Checks (Immediate Reinforcement)
1. **Program Structure:** Design appropriate program organization
2. **Argument Parsing:** Create robust CLI interfaces
3. **Path Handling:** Choose appropriate path manipulation methods
4. **Error Handling:** Implement proper error handling for system operations

### Try This Exercises (Hands-on Application)
1. **Script Creation:** Build standalone Python utilities
2. **CLI Development:** Create command-line interfaces with argparse
3. **File Operations:** Implement file processing utilities
4. **Directory Management:** Build directory manipulation tools

### Lab Problems (Critical Thinking)
1. **File Organizer:** Create a utility to organize files by type/date
2. **Log Analyzer:** Build a command-line log analysis tool
3. **Backup System:** Implement a simple backup utility
4. **System Monitor:** Create a system information gathering tool

## Assessment Criteria
- **Program Structure:** Well-organized, maintainable code structure
- **CLI Design:** User-friendly command-line interfaces
- **Error Handling:** Robust error handling and user feedback
- **Cross-platform Compatibility:** Code that works across different operating systems
- **Documentation:** Clear usage instructions and help systems
- **Resource Management:** Proper cleanup and resource handling

## Resources
- [Python argparse Documentation](https://docs.python.org/3/library/argparse.html)
- [pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
- [os Module Documentation](https://docs.python.org/3/library/os.html)
- [shutil Module Documentation](https://docs.python.org/3/library/shutil.html)
- [Python Scripting Best Practices](https://docs.python.org/3/tutorial/stdlib.html)

### Additional Topics
- `fileinput` module for stream processing from stdin/files.
- Packaging overview: `zipapp` for single-file apps; building wheels for distribution.

### AI-comparison callout
For CLI and filesystem labs, compare AI code for:
- Efficient streaming vs full-file reads
- Correct argparse defaults and help text
- Safe file operations and error handling

## Next Steps
After completing Module 4, you'll be ready to explore Object-Oriented Programming and advanced language features in Module 5. The program structure and system interaction skills you develop here will be essential for building larger applications.
