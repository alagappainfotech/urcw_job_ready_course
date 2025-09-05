# Module 9: File Handling and I/O Operations - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master file operations and I/O handling in Python
- Work with different file formats (text, binary, CSV, JSON, XML)
- Implement robust file processing pipelines
- Handle file compression and archiving
- Work with file system operations and path manipulation
- Build efficient file monitoring and processing systems
- Implement secure file handling practices

## Core Concepts

### 1. Basic File Operations

#### File Reading and Writing
```python
# Basic file operations
def read_file_basic(filename: str) -> str:
    """Read entire file content"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return ""
    except PermissionError:
        print(f"Permission denied to read {filename}")
        return ""

def write_file_basic(filename: str, content: str) -> bool:
    """Write content to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    except PermissionError:
        print(f"Permission denied to write {filename}")
        return False
    except OSError as e:
        print(f"OS error: {e}")
        return False

def append_to_file(filename: str, content: str) -> bool:
    """Append content to file"""
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error appending to file: {e}")
        return False

# Line-by-line reading
def read_file_lines(filename: str) -> list:
    """Read file line by line"""
    lines = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.rstrip('\n'))
    except Exception as e:
        print(f"Error reading file: {e}")
    return lines

def read_file_generator(filename: str):
    """Generator for reading large files efficiently"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.rstrip('\n')
    except Exception as e:
        print(f"Error reading file: {e}")
        return
```

#### Binary File Operations
```python
def read_binary_file(filename: str) -> bytes:
    """Read binary file"""
    try:
        with open(filename, 'rb') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading binary file: {e}")
        return b""

def write_binary_file(filename: str, data: bytes) -> bool:
    """Write binary data to file"""
    try:
        with open(filename, 'wb') as file:
            file.write(data)
        return True
    except Exception as e:
        print(f"Error writing binary file: {e}")
        return False

def copy_file(source: str, destination: str) -> bool:
    """Copy file from source to destination"""
    try:
        with open(source, 'rb') as src, open(destination, 'wb') as dst:
            while True:
                chunk = src.read(8192)  # Read in chunks
                if not chunk:
                    break
                dst.write(chunk)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False
```

### 2. File System Operations

#### Path Manipulation with pathlib
```python
from pathlib import Path
import os
import shutil

class FileSystemManager:
    """Advanced file system operations using pathlib"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    def create_directory(self, dir_path: str) -> bool:
        """Create directory and parent directories if needed"""
        try:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    def delete_directory(self, dir_path: str) -> bool:
        """Delete directory and all contents"""
        try:
            full_path = self.base_path / dir_path
            if full_path.exists():
                shutil.rmtree(full_path)
            return True
        except Exception as e:
            print(f"Error deleting directory: {e}")
            return False
    
    def list_files(self, pattern: str = "*") -> list:
        """List files matching pattern"""
        try:
            return list(self.base_path.glob(pattern))
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def find_files_by_extension(self, extension: str) -> list:
        """Find all files with specific extension"""
        pattern = f"**/*.{extension}"
        return self.list_files(pattern)
    
    def get_file_info(self, file_path: str) -> dict:
        """Get detailed file information"""
        try:
            full_path = self.base_path / file_path
            if not full_path.exists():
                return {}
            
            stat = full_path.stat()
            return {
                'name': full_path.name,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_file': full_path.is_file(),
                'is_dir': full_path.is_dir(),
                'extension': full_path.suffix,
                'parent': str(full_path.parent)
            }
        except Exception as e:
            print(f"Error getting file info: {e}")
            return {}
    
    def move_file(self, source: str, destination: str) -> bool:
        """Move file from source to destination"""
        try:
            src_path = self.base_path / source
            dst_path = self.base_path / destination
            
            # Create destination directory if it doesn't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dst_path))
            return True
        except Exception as e:
            print(f"Error moving file: {e}")
            return False
    
    def copy_file(self, source: str, destination: str) -> bool:
        """Copy file from source to destination"""
        try:
            src_path = self.base_path / source
            dst_path = self.base_path / destination
            
            # Create destination directory if it doesn't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(str(src_path), str(dst_path))
            return True
        except Exception as e:
            print(f"Error copying file: {e}")
            return False
```

### 3. File Format Handling

#### CSV File Operations
```python
import csv
from typing import List, Dict, Any

class CSVHandler:
    """CSV file reading and writing utilities"""
    
    def read_csv(self, filename: str, delimiter: str = ',') -> List[Dict[str, Any]]:
        """Read CSV file and return list of dictionaries"""
        data = []
        try:
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=delimiter)
                for row in reader:
                    data.append(dict(row))
        except Exception as e:
            print(f"Error reading CSV: {e}")
        return data
    
    def write_csv(self, filename: str, data: List[Dict[str, Any]], 
                  fieldnames: List[str] = None) -> bool:
        """Write data to CSV file"""
        try:
            if not data:
                return False
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Error writing CSV: {e}")
            return False
    
    def append_csv(self, filename: str, data: List[Dict[str, Any]], 
                   fieldnames: List[str] = None) -> bool:
        """Append data to existing CSV file"""
        try:
            if not data:
                return False
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            file_exists = Path(filename).exists()
            
            with open(filename, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Error appending CSV: {e}")
            return False
```

#### JSON File Operations
```python
import json
from typing import Any, Union

class JSONHandler:
    """JSON file reading and writing utilities"""
    
    def read_json(self, filename: str) -> Union[dict, list, None]:
        """Read JSON file and return Python object"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File {filename} not found")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return None
    
    def write_json(self, filename: str, data: Any, indent: int = 2) -> bool:
        """Write Python object to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing JSON: {e}")
            return False
    
    def update_json(self, filename: str, updates: dict) -> bool:
        """Update specific keys in JSON file"""
        try:
            data = self.read_json(filename)
            if data is None:
                return False
            
            if isinstance(data, dict):
                data.update(updates)
                return self.write_json(filename, data)
            return False
        except Exception as e:
            print(f"Error updating JSON: {e}")
            return False
```

#### XML File Operations
```python
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

class XMLHandler:
    """XML file reading and writing utilities"""
    
    def read_xml(self, filename: str) -> ET.Element:
        """Read XML file and return root element"""
        try:
            tree = ET.parse(filename)
            return tree.getroot()
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return None
        except Exception as e:
            print(f"Error reading XML: {e}")
            return None
    
    def write_xml(self, filename: str, root: ET.Element) -> bool:
        """Write XML element to file"""
        try:
            tree = ET.ElementTree(root)
            tree.write(filename, encoding='utf-8', xml_declaration=True)
            return True
        except Exception as e:
            print(f"Error writing XML: {e}")
            return False
    
    def xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result['text'] = element.text.strip()
        
        # Add children
        children = {}
        for child in element:
            child_dict = self.xml_to_dict(child)
            if child.tag in children:
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_dict)
            else:
                children[child.tag] = child_dict
        
        if children:
            result.update(children)
        
        return result
    
    def dict_to_xml(self, data: Dict[str, Any], root_tag: str = 'root') -> ET.Element:
        """Convert dictionary to XML element"""
        root = ET.Element(root_tag)
        self._dict_to_xml_recursive(data, root)
        return root
    
    def _dict_to_xml_recursive(self, data: Any, parent: ET.Element):
        """Recursively convert dictionary to XML"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == '@attributes':
                    parent.attrib.update(value)
                elif key == 'text':
                    parent.text = str(value)
                else:
                    child = ET.SubElement(parent, key)
                    self._dict_to_xml_recursive(value, child)
        elif isinstance(data, list):
            for item in data:
                self._dict_to_xml_recursive(item, parent)
        else:
            parent.text = str(data)
```

### 4. File Compression and Archiving

#### ZIP File Operations
```python
import zipfile
import tarfile
from pathlib import Path

class ArchiveManager:
    """File compression and archiving utilities"""
    
    def create_zip(self, zip_filename: str, files_to_compress: List[str]) -> bool:
        """Create ZIP archive from list of files"""
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files_to_compress:
                    if Path(file_path).exists():
                        zipf.write(file_path, Path(file_path).name)
            return True
        except Exception as e:
            print(f"Error creating ZIP: {e}")
            return False
    
    def extract_zip(self, zip_filename: str, extract_to: str = ".") -> bool:
        """Extract ZIP archive"""
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zipf:
                zipf.extractall(extract_to)
            return True
        except Exception as e:
            print(f"Error extracting ZIP: {e}")
            return False
    
    def list_zip_contents(self, zip_filename: str) -> List[str]:
        """List contents of ZIP archive"""
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zipf:
                return zipf.namelist()
        except Exception as e:
            print(f"Error listing ZIP contents: {e}")
            return []
    
    def create_tar(self, tar_filename: str, files_to_compress: List[str], 
                   compression: str = 'gz') -> bool:
        """Create TAR archive with optional compression"""
        try:
            mode = f'w:{compression}' if compression else 'w'
            with tarfile.open(tar_filename, mode) as tar:
                for file_path in files_to_compress:
                    if Path(file_path).exists():
                        tar.add(file_path, arcname=Path(file_path).name)
            return True
        except Exception as e:
            print(f"Error creating TAR: {e}")
            return False
    
    def extract_tar(self, tar_filename: str, extract_to: str = ".") -> bool:
        """Extract TAR archive"""
        try:
            with tarfile.open(tar_filename, 'r:*') as tar:
                tar.extractall(extract_to)
            return True
        except Exception as e:
            print(f"Error extracting TAR: {e}")
            return False
```

### 5. File Monitoring and Processing

#### File Watcher
```python
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileWatcher(FileSystemEventHandler):
    """Monitor file system changes"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.observer = Observer()
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            if self.callback:
                self.callback(event.src_path, 'modified')
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            print(f"File created: {event.src_path}")
            if self.callback:
                self.callback(event.src_path, 'created')
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            if self.callback:
                self.callback(event.src_path, 'deleted')
    
    def start_watching(self, directory: str):
        """Start watching directory for changes"""
        self.observer.schedule(self, directory, recursive=True)
        self.observer.start()
        print(f"Watching directory: {directory}")
    
    def stop_watching(self):
        """Stop watching for changes"""
        self.observer.stop()
        self.observer.join()
```

#### Batch File Processor
```python
class BatchFileProcessor:
    """Process multiple files in batches"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.processed_files = []
        self.failed_files = []
    
    def process_files(self, file_list: List[str], processor_func) -> Dict[str, Any]:
        """Process files in batches"""
        results = {
            'processed': 0,
            'failed': 0,
            'total': len(file_list)
        }
        
        for i in range(0, len(file_list), self.batch_size):
            batch = file_list[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}: {len(batch)} files")
            
            for file_path in batch:
                try:
                    result = processor_func(file_path)
                    self.processed_files.append(file_path)
                    results['processed'] += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    self.failed_files.append(file_path)
                    results['failed'] += 1
        
        return results
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        return {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'success_rate': len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100
        }
```

### 6. Advanced File Operations

#### File Encryption and Security
```python
import hashlib
import base64
from cryptography.fernet import Fernet

class FileSecurity:
    """File encryption and security utilities"""
    
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt_file(self, input_file: str, output_file: str) -> bool:
        """Encrypt file"""
        try:
            with open(input_file, 'rb') as file:
                data = file.read()
            
            encrypted_data = self.cipher.encrypt(data)
            
            with open(output_file, 'wb') as file:
                file.write(encrypted_data)
            
            return True
        except Exception as e:
            print(f"Error encrypting file: {e}")
            return False
    
    def decrypt_file(self, input_file: str, output_file: str) -> bool:
        """Decrypt file"""
        try:
            with open(input_file, 'rb') as file:
                encrypted_data = file.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            with open(output_file, 'wb') as file:
                file.write(decrypted_data)
            
            return True
        except Exception as e:
            print(f"Error decrypting file: {e}")
            return False
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate file hash"""
        try:
            hash_func = hashlib.new(algorithm)
            with open(file_path, 'rb') as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return ""
    
    def verify_file_integrity(self, file_path: str, expected_hash: str, 
                             algorithm: str = 'sha256') -> bool:
        """Verify file integrity using hash"""
        actual_hash = self.calculate_file_hash(file_path, algorithm)
        return actual_hash == expected_hash
```

## Best Practices

### 1. Error Handling and Resource Management
```python
class SafeFileHandler:
    """Safe file handling with proper error management"""
    
    def __init__(self):
        self.open_files = []
    
    def safe_open(self, filename: str, mode: str = 'r', encoding: str = 'utf-8'):
        """Safely open file with proper error handling"""
        try:
            file = open(filename, mode, encoding=encoding)
            self.open_files.append(file)
            return file
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
        except PermissionError:
            print(f"Permission denied: {filename}")
            return None
        except Exception as e:
            print(f"Error opening file {filename}: {e}")
            return None
    
    def safe_close_all(self):
        """Close all open files"""
        for file in self.open_files:
            try:
                file.close()
            except Exception as e:
                print(f"Error closing file: {e}")
        self.open_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.safe_close_all()
```

### 2. Performance Optimization
```python
class OptimizedFileProcessor:
    """Optimized file processing for large files"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
    
    def process_large_file(self, filename: str, processor_func) -> List[Any]:
        """Process large file in chunks"""
        results = []
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                while True:
                    chunk = file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    # Process chunk
                    chunk_results = processor_func(chunk)
                    results.extend(chunk_results)
        except Exception as e:
            print(f"Error processing large file: {e}")
        
        return results
    
    def parallel_file_processing(self, file_list: List[str], processor_func, 
                               max_workers: int = 4) -> List[Any]:
        """Process multiple files in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(processor_func, file_path) for file_path in file_list]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    results.append(None)
        
        return results
```

## Quick Checks

### Check 1: File Reading
```python
# What will this code do?
with open("test.txt", "w") as f:
    f.write("Hello World")
    
with open("test.txt", "r") as f:
    content = f.read()
print(content)
```

### Check 2: Path Operations
```python
# What will this return?
from pathlib import Path
path = Path("/home/user/documents/file.txt")
print(path.name)
print(path.suffix)
print(path.parent)
```

### Check 3: CSV Handling
```python
# What will this create?
import csv
data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
with open("test.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()
    writer.writerows(data)
```

## Lab Problems

### Lab 1: File Management System
Build a comprehensive file management system that can organize, search, and process files based on various criteria.

### Lab 2: Data Processing Pipeline
Create a data processing pipeline that can handle multiple file formats and convert between them efficiently.

### Lab 3: File Backup System
Implement an automated file backup system with compression, encryption, and scheduling capabilities.

### Lab 4: Log File Analyzer
Build a log file analyzer that can process large log files, extract patterns, and generate reports.

## AI Code Comparison
When working with AI-generated file handling code, evaluate:
- **Error handling** - are all possible exceptions properly caught and handled?
- **Resource management** - are files properly closed and resources released?
- **Security considerations** - are file operations secure and validated?
- **Performance** - is the code optimized for large files and efficient processing?
- **Cross-platform compatibility** - does the code work on different operating systems?

## Next Steps
- Learn about advanced file formats and data serialization
- Master database operations and ORM frameworks
- Explore cloud storage and distributed file systems
- Study file system monitoring and automation
- Understand file security and encryption best practices
