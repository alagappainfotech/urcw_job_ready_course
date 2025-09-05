# Module 10: Regular Expressions and Pattern Matching - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master regular expressions for complex pattern matching
- Use regex for text validation, extraction, and manipulation
- Implement efficient pattern matching algorithms
- Build robust text processing pipelines with regex
- Handle edge cases and optimize regex performance
- Apply regex in real-world data processing scenarios
- Debug and troubleshoot regex patterns effectively

## Core Concepts

### 1. Regular Expression Fundamentals

#### Basic Pattern Matching
```python
import re

# Basic pattern matching
def basic_pattern_matching():
    text = "The quick brown fox jumps over the lazy dog"
    
    # Simple search
    pattern = r"fox"
    match = re.search(pattern, text)
    if match:
        print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")
    
    # Case-insensitive search
    pattern = re.compile(r"FOX", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        print(f"Found '{match.group()}' (case-insensitive)")
    
    # Find all matches
    pattern = r"o"
    matches = re.findall(pattern, text)
    print(f"Found 'o' {len(matches)} times: {matches}")
    
    # Find all matches with positions
    for match in re.finditer(r"o", text):
        print(f"Found 'o' at position {match.start()}")

# Character classes and quantifiers
def character_classes_and_quantifiers():
    text = "The quick brown fox jumps over the lazy dog"
    
    # Character classes
    vowels = re.findall(r"[aeiou]", text)
    print(f"Vowels found: {vowels}")
    
    # Negated character class
    consonants = re.findall(r"[^aeiou\s]", text)
    print(f"Consonants found: {consonants}")
    
    # Predefined character classes
    digits = re.findall(r"\d", "Phone: 123-456-7890")
    print(f"Digits found: {digits}")
    
    word_chars = re.findall(r"\w", text)
    print(f"Word characters found: {len(word_chars)}")
    
    # Quantifiers
    text_with_repeats = "aaabbbccc"
    
    # Zero or more
    zero_or_more = re.findall(r"a*", text_with_repeats)
    print(f"Zero or more 'a's: {zero_or_more}")
    
    # One or more
    one_or_more = re.findall(r"a+", text_with_repeats)
    print(f"One or more 'a's: {one_or_more}")
    
    # Exactly 3
    exactly_three = re.findall(r"a{3}", text_with_repeats)
    print(f"Exactly 3 'a's: {exactly_three}")
    
    # Between 2 and 4
    between_two_four = re.findall(r"a{2,4}", text_with_repeats)
    print(f"Between 2 and 4 'a's: {between_two_four}")
```

#### Anchors and Boundaries
```python
def anchors_and_boundaries():
    text = "Start of line\nMiddle of line\nEnd of line"
    
    # Line start anchor
    start_matches = re.findall(r"^Start", text, re.MULTILINE)
    print(f"Lines starting with 'Start': {start_matches}")
    
    # Line end anchor
    end_matches = re.findall(r"line$", text, re.MULTILINE)
    print(f"Lines ending with 'line': {end_matches}")
    
    # Word boundaries
    text_with_words = "The quick brown fox jumps over the lazy dog"
    word_matches = re.findall(r"\b\w{4}\b", text_with_words)  # 4-letter words
    print(f"4-letter words: {word_matches}")
    
    # Non-word boundaries
    non_word_matches = re.findall(r"\B\w{2}\B", text_with_words)  # 2-letter substrings
    print(f"2-letter substrings: {non_word_matches}")
```

### 2. Advanced Pattern Matching

#### Groups and Capturing
```python
def groups_and_capturing():
    text = "John Doe, Jane Smith, Bob Johnson"
    
    # Simple groups
    pattern = r"(\w+)\s+(\w+)"  # Capture first and last names
    matches = re.findall(pattern, text)
    print(f"Name pairs: {matches}")
    
    # Named groups
    pattern = r"(?P<first>\w+)\s+(?P<last>\w+)"
    match = re.search(pattern, text)
    if match:
        print(f"First name: {match.group('first')}")
        print(f"Last name: {match.group('last')}")
    
    # Non-capturing groups
    text_with_colors = "color colour"
    pattern = r"colou(?:r)?"  # Non-capturing group
    matches = re.findall(pattern, text_with_colors)
    print(f"Color matches (non-capturing): {matches}")
    
    # Backreferences
    text_with_repeats = "hello hello world world"
    pattern = r"(\w+)\s+\1"  # Match repeated words
    matches = re.findall(pattern, text_with_repeats)
    print(f"Repeated words: {matches}")
    
    # Complex grouping example
    phone_pattern = r"(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
    phone_text = "Call us at (555) 123-4567 or 555-123-4567"
    phone_matches = re.findall(phone_pattern, phone_text)
    print(f"Phone numbers: {phone_matches}")
```

#### Lookahead and Lookbehind
```python
def lookahead_and_lookbehind():
    text = "The price is $100.50 and the total is $200.75"
    
    # Positive lookahead - find numbers followed by decimal point
    lookahead_matches = re.findall(r"\d+(?=\.)", text)
    print(f"Numbers before decimal: {lookahead_matches}")
    
    # Negative lookahead - find numbers not followed by decimal point
    negative_lookahead = re.findall(r"\d+(?!\.)", text)
    print(f"Numbers not before decimal: {negative_lookahead}")
    
    # Positive lookbehind - find numbers preceded by dollar sign
    lookbehind_matches = re.findall(r"(?<=\$)\d+\.?\d*", text)
    print(f"Numbers after $: {lookbehind_matches}")
    
    # Negative lookbehind - find numbers not preceded by dollar sign
    negative_lookbehind = re.findall(r"(?<!\$)\d+", text)
    print(f"Numbers not after $: {negative_lookbehind}")
    
    # Complex example - find words that are not followed by "ing"
    text_with_ing = "running jumping walk talk"
    not_ing_matches = re.findall(r"\w+(?!ing)\b", text_with_ing)
    print(f"Words not ending with 'ing': {not_ing_matches}")
```

### 3. Text Validation and Extraction

#### Email and URL Validation
```python
class TextValidator:
    """Advanced text validation using regex"""
    
    def __init__(self):
        # Email pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Phone number pattern
        self.phone_pattern = re.compile(
            r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        )
        
        # Credit card pattern
        self.credit_card_pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        )
        
        # SSN pattern
        self.ssn_pattern = re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        )
    
    def validate_email(self, email: str) -> bool:
        """Validate email address"""
        return bool(self.email_pattern.match(email))
    
    def validate_url(self, url: str) -> bool:
        """Validate URL"""
        return bool(self.url_pattern.match(url))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number"""
        return bool(self.phone_pattern.match(phone))
    
    def extract_emails(self, text: str) -> list:
        """Extract all email addresses from text"""
        return self.email_pattern.findall(text)
    
    def extract_urls(self, text: str) -> list:
        """Extract all URLs from text"""
        return self.url_pattern.findall(text)
    
    def extract_phone_numbers(self, text: str) -> list:
        """Extract all phone numbers from text"""
        matches = self.phone_pattern.findall(text)
        return [''.join(match) for match in matches]
    
    def extract_credit_cards(self, text: str) -> list:
        """Extract credit card numbers from text"""
        return self.credit_card_pattern.findall(text)
    
    def extract_ssns(self, text: str) -> list:
        """Extract SSNs from text"""
        return self.ssn_pattern.findall(text)
```

#### Data Extraction and Parsing
```python
class DataExtractor:
    """Extract structured data from text using regex"""
    
    def __init__(self):
        # Date patterns
        self.date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
        ]
        
        # Time patterns
        self.time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b',  # HH:MM:SS AM/PM
        ]
        
        # IP address pattern
        self.ip_pattern = re.compile(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        )
        
        # MAC address pattern
        self.mac_pattern = re.compile(
            r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
        )
    
    def extract_dates(self, text: str) -> list:
        """Extract dates from text"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates
    
    def extract_times(self, text: str) -> list:
        """Extract times from text"""
        times = []
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text)
            times.extend(matches)
        return times
    
    def extract_ip_addresses(self, text: str) -> list:
        """Extract IP addresses from text"""
        return self.ip_pattern.findall(text)
    
    def extract_mac_addresses(self, text: str) -> list:
        """Extract MAC addresses from text"""
        return self.mac_pattern.findall(text)
    
    def extract_hashtags(self, text: str) -> list:
        """Extract hashtags from text"""
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, text)
    
    def extract_mentions(self, text: str) -> list:
        """Extract mentions from text"""
        mention_pattern = r'@\w+'
        return re.findall(mention_pattern, text)
    
    def extract_html_tags(self, text: str) -> list:
        """Extract HTML tags from text"""
        html_pattern = r'<[^>]+>'
        return re.findall(html_pattern, text)
    
    def extract_currency(self, text: str) -> list:
        """Extract currency amounts from text"""
        currency_pattern = r'\$[\d,]+\.?\d*'
        return re.findall(currency_pattern, text)
```

### 4. Text Manipulation and Replacement

#### Advanced Text Replacement
```python
class TextManipulator:
    """Advanced text manipulation using regex"""
    
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return self.punctuation_pattern.sub('', text)
    
    def extract_words(self, text: str) -> list:
        """Extract all words from text"""
        return self.word_pattern.findall(text)
    
    def replace_words(self, text: str, word_map: dict) -> str:
        """Replace words using dictionary mapping"""
        def replace_func(match):
            word = match.group()
            return word_map.get(word.lower(), word)
        
        return self.word_pattern.sub(replace_func, text)
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        # Mask credit card numbers
        text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '****-****-****-****', text)
        
        # Mask SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****', text)
        
        # Mask email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '***@***.***', text)
        
        return text
    
    def format_phone_numbers(self, text: str, format_style: str = 'standard') -> str:
        """Format phone numbers in text"""
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        
        def format_phone(match):
            country_code, area, exchange, number = match.groups()
            
            if format_style == 'standard':
                return f"({area}) {exchange}-{number}"
            elif format_style == 'international':
                return f"+1-{area}-{exchange}-{number}"
            elif format_style == 'dots':
                return f"{area}.{exchange}.{number}"
            else:
                return f"{area}{exchange}{number}"
        
        return re.sub(phone_pattern, format_phone, text)
    
    def extract_and_replace(self, text: str, pattern: str, replacement_func) -> tuple:
        """Extract matches and replace them using a function"""
        matches = []
        
        def replace_func(match):
            matches.append(match.group())
            return replacement_func(match.group())
        
        new_text = re.sub(pattern, replace_func, text)
        return new_text, matches
```

### 5. Performance Optimization

#### Compiled Patterns and Caching
```python
import functools
from typing import Pattern

class OptimizedRegex:
    """Optimized regex operations with caching"""
    
    def __init__(self):
        self._compiled_patterns = {}
    
    def get_compiled_pattern(self, pattern: str, flags: int = 0) -> Pattern:
        """Get compiled pattern with caching"""
        key = (pattern, flags)
        if key not in self._compiled_patterns:
            self._compiled_patterns[key] = re.compile(pattern, flags)
        return self._compiled_patterns[key]
    
    @functools.lru_cache(maxsize=128)
    def cached_search(self, pattern: str, text: str, flags: int = 0) -> list:
        """Cached regex search"""
        compiled_pattern = self.get_compiled_pattern(pattern, flags)
        return compiled_pattern.findall(text)
    
    def batch_process(self, patterns: list, text: str) -> dict:
        """Process multiple patterns efficiently"""
        results = {}
        for pattern in patterns:
            compiled_pattern = self.get_compiled_pattern(pattern)
            results[pattern] = compiled_pattern.findall(text)
        return results
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better regex performance"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = re.sub(r'\r\n|\r', '\n', text)
        
        return text.strip()
```

#### Memory-Efficient Processing
```python
class MemoryEfficientRegex:
    """Memory-efficient regex processing for large texts"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
    
    def process_large_file(self, filename: str, pattern: str, 
                          processor_func) -> list:
        """Process large file in chunks"""
        results = []
        compiled_pattern = re.compile(pattern)
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                buffer = ""
                while True:
                    chunk = file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # Process complete matches
                    matches = list(compiled_pattern.finditer(buffer))
                    for match in matches:
                        results.append(processor_func(match))
                    
                    # Keep incomplete match at end of buffer
                    if matches:
                        last_match = matches[-1]
                        buffer = buffer[last_match.end():]
                    else:
                        # Keep last part of buffer for potential matches
                        buffer = buffer[-len(pattern):]
        except Exception as e:
            print(f"Error processing large file: {e}")
        
        return results
    
    def stream_process(self, text_stream, pattern: str, processor_func):
        """Process text stream with regex"""
        compiled_pattern = re.compile(pattern)
        
        for line in text_stream:
            for match in compiled_pattern.finditer(line):
                yield processor_func(match)
```

### 6. Real-World Applications

#### Log File Processing
```python
class LogProcessor:
    """Process log files using regex"""
    
    def __init__(self):
        # Common log patterns
        self.log_patterns = {
            'apache': r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)',
            'nginx': r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)"',
            'syslog': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}) (\w+) (.*)',
        }
    
    def parse_apache_log(self, log_line: str) -> dict:
        """Parse Apache log line"""
        pattern = self.log_patterns['apache']
        match = re.match(pattern, log_line)
        
        if match:
            return {
                'ip': match.group(1),
                'timestamp': match.group(2),
                'request': match.group(3),
                'status': int(match.group(4)),
                'size': int(match.group(5))
            }
        return {}
    
    def extract_error_codes(self, log_file: str) -> dict:
        """Extract error codes from log file"""
        error_counts = {}
        
        try:
            with open(log_file, 'r') as file:
                for line in file:
                    match = re.search(r'" (\d{3}) ', line)
                    if match:
                        status_code = match.group(1)
                        error_counts[status_code] = error_counts.get(status_code, 0) + 1
        except Exception as e:
            print(f"Error processing log file: {e}")
        
        return error_counts
    
    def find_suspicious_activity(self, log_file: str) -> list:
        """Find suspicious activity in log file"""
        suspicious_patterns = [
            r'admin.*login',
            r'failed.*password',
            r'sql.*injection',
            r'xss.*attack',
            r'brute.*force'
        ]
        
        suspicious_activities = []
        
        try:
            with open(log_file, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    for pattern in suspicious_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            suspicious_activities.append({
                                'line_number': line_num,
                                'pattern': pattern,
                                'line': line.strip()
                            })
        except Exception as e:
            print(f"Error processing log file: {e}")
        
        return suspicious_activities
```

#### Data Cleaning and Validation
```python
class DataCleaner:
    """Clean and validate data using regex"""
    
    def __init__(self):
        self.cleaners = {
            'email': self._clean_email,
            'phone': self._clean_phone,
            'name': self._clean_name,
            'address': self._clean_address,
            'date': self._clean_date
        }
    
    def _clean_email(self, email: str) -> str:
        """Clean email address"""
        # Remove extra whitespace
        email = email.strip()
        
        # Convert to lowercase
        email = email.lower()
        
        # Remove invalid characters
        email = re.sub(r'[^\w@.-]', '', email)
        
        return email
    
    def _clean_phone(self, phone: str) -> str:
        """Clean phone number"""
        # Remove all non-digit characters except +
        phone = re.sub(r'[^\d+]', '', phone)
        
        # Add country code if missing
        if not phone.startswith('+') and len(phone) == 10:
            phone = '+1' + phone
        
        return phone
    
    def _clean_name(self, name: str) -> str:
        """Clean name"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def _clean_address(self, address: str) -> str:
        """Clean address"""
        # Remove extra whitespace
        address = re.sub(r'\s+', ' ', address.strip())
        
        # Standardize common abbreviations
        abbreviations = {
            r'\bSt\b': 'Street',
            r'\bAve\b': 'Avenue',
            r'\bRd\b': 'Road',
            r'\bBlvd\b': 'Boulevard',
            r'\bDr\b': 'Drive',
            r'\bCt\b': 'Court',
            r'\bLn\b': 'Lane'
        }
        
        for pattern, replacement in abbreviations.items():
            address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
        
        return address
    
    def _clean_date(self, date: str) -> str:
        """Clean date"""
        # Remove extra whitespace
        date = date.strip()
        
        # Standardize separators
        date = re.sub(r'[-/]', '/', date)
        
        return date
    
    def clean_data(self, data: dict, fields_to_clean: list = None) -> dict:
        """Clean data dictionary"""
        if fields_to_clean is None:
            fields_to_clean = list(data.keys())
        
        cleaned_data = {}
        
        for field, value in data.items():
            if field in fields_to_clean and field in self.cleaners:
                cleaned_data[field] = self.cleaners[field](str(value))
            else:
                cleaned_data[field] = value
        
        return cleaned_data
```

## Best Practices

### 1. Regex Debugging and Testing
```python
class RegexTester:
    """Test and debug regex patterns"""
    
    def __init__(self):
        self.test_cases = []
    
    def add_test_case(self, pattern: str, text: str, expected: bool, description: str = ""):
        """Add test case for regex pattern"""
        self.test_cases.append({
            'pattern': pattern,
            'text': text,
            'expected': expected,
            'description': description
        })
    
    def run_tests(self) -> dict:
        """Run all test cases"""
        results = {
            'passed': 0,
            'failed': 0,
            'total': len(self.test_cases),
            'failures': []
        }
        
        for test_case in self.test_cases:
            try:
                pattern = re.compile(test_case['pattern'])
                actual = bool(pattern.search(test_case['text']))
                
                if actual == test_case['expected']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['failures'].append({
                        'pattern': test_case['pattern'],
                        'text': test_case['text'],
                        'expected': test_case['expected'],
                        'actual': actual,
                        'description': test_case['description']
                    })
            except re.error as e:
                results['failed'] += 1
                results['failures'].append({
                    'pattern': test_case['pattern'],
                    'text': test_case['text'],
                    'expected': test_case['expected'],
                    'actual': f"Regex Error: {e}",
                    'description': test_case['description']
                })
        
        return results
    
    def debug_pattern(self, pattern: str, text: str) -> dict:
        """Debug regex pattern with detailed information"""
        try:
            compiled_pattern = re.compile(pattern)
            match = compiled_pattern.search(text)
            
            return {
                'pattern': pattern,
                'text': text,
                'match_found': bool(match),
                'match_groups': match.groups() if match else [],
                'match_span': match.span() if match else None,
                'full_match': match.group() if match else None
            }
        except re.error as e:
            return {
                'pattern': pattern,
                'text': text,
                'error': str(e)
            }
```

### 2. Performance Monitoring
```python
import time
import cProfile
import pstats
from io import StringIO

class RegexProfiler:
    """Profile regex performance"""
    
    def __init__(self):
        self.results = {}
    
    def profile_pattern(self, pattern: str, text: str, iterations: int = 1000) -> dict:
        """Profile regex pattern performance"""
        compiled_pattern = re.compile(pattern)
        
        # Time the pattern
        start_time = time.time()
        for _ in range(iterations):
            compiled_pattern.findall(text)
        end_time = time.time()
        
        # Profile with cProfile
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(iterations):
            compiled_pattern.findall(text)
        pr.disable()
        
        # Get profiling results
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        return {
            'pattern': pattern,
            'text_length': len(text),
            'iterations': iterations,
            'total_time': end_time - start_time,
            'avg_time_per_iteration': (end_time - start_time) / iterations,
            'profile_output': s.getvalue()
        }
    
    def compare_patterns(self, patterns: list, text: str, iterations: int = 1000) -> dict:
        """Compare performance of multiple patterns"""
        results = {}
        
        for pattern in patterns:
            results[pattern] = self.profile_pattern(pattern, text, iterations)
        
        return results
```

## Quick Checks

### Check 1: Basic Pattern Matching
```python
# What will this print?
import re
text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"
match = re.search(pattern, text)
print(match.group() if match else "No match")
```

### Check 2: Character Classes
```python
# What will this return?
import re
text = "Hello123World456"
pattern = r"[0-9]+"
matches = re.findall(pattern, text)
print(matches)
```

### Check 3: Groups and Capturing
```python
# What will this return?
import re
text = "John Doe, Jane Smith"
pattern = r"(\w+)\s+(\w+)"
matches = re.findall(pattern, text)
print(matches)
```

## Lab Problems

### Lab 1: Email Validator
Build a comprehensive email validation system that can handle various email formats and edge cases.

### Lab 2: Log File Analyzer
Create a log file analyzer that can extract specific information from different log formats using regex.

### Lab 3: Data Extraction Tool
Build a tool that can extract structured data from unstructured text using regex patterns.

### Lab 4: Text Sanitizer
Implement a text sanitization tool that can clean and normalize text data using regex.

## AI Code Comparison
When working with AI-generated regex code, evaluate:
- **Pattern accuracy** - does the regex correctly match the intended patterns?
- **Performance** - is the regex optimized for the use case?
- **Edge cases** - are boundary conditions and special cases handled?
- **Readability** - is the regex pattern clear and well-documented?
- **Security** - are there any potential security vulnerabilities in the regex?

## Next Steps
- Learn about advanced regex features and optimization techniques
- Master regex in different programming languages and tools
- Explore regex in data processing and ETL pipelines
- Study regex performance optimization and debugging
- Understand regex security considerations and best practices
