"""
Module 10: Regular Expressions and Pattern Matching - Exercises
Complete these exercises to master regular expressions and pattern matching.
"""

import re
import time
import functools
from typing import List, Dict, Any, Optional, Pattern, Tuple
from collections import defaultdict
import cProfile
import pstats
from io import StringIO

# Exercise 1: Basic Pattern Matching
class BasicPatternMatcher:
    """Basic pattern matching utilities"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
            'html_tag': r'<[^>]+>',
            'whitespace': r'\s+',
            'word_boundary': r'\b\w+\b',
            'sentence': r'[.!?]+',
            'paragraph': r'\n\s*\n'
        }
    
    def find_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Find all matches for a specific pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return re.findall(self.patterns[pattern_name], text)
    
    def replace_pattern(self, text: str, pattern_name: str, replacement: str) -> str:
        """Replace all matches of a pattern with replacement"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return re.sub(self.patterns[pattern_name], replacement, text)
    
    def split_by_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Split text by a pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return re.split(self.patterns[pattern_name], text)
    
    def validate_pattern(self, text: str, pattern_name: str) -> bool:
        """Validate if text matches pattern exactly"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return bool(re.fullmatch(self.patterns[pattern_name], text))

# Exercise 2: Advanced Pattern Matching
class AdvancedPatternMatcher:
    """Advanced pattern matching with groups and lookahead/lookbehind"""
    
    def __init__(self):
        self.compiled_patterns = {}
    
    def compile_pattern(self, pattern: str, flags: int = 0) -> Pattern:
        """Compile pattern with caching"""
        key = (pattern, flags)
        if key not in self.compiled_patterns:
            self.compiled_patterns[key] = re.compile(pattern, flags)
        return self.compiled_patterns[key]
    
    def extract_groups(self, text: str, pattern: str) -> List[Tuple[str, ...]]:
        """Extract groups from text using pattern"""
        compiled_pattern = self.compile_pattern(pattern)
        matches = compiled_pattern.findall(text)
        return matches
    
    def extract_named_groups(self, text: str, pattern: str) -> List[Dict[str, str]]:
        """Extract named groups from text"""
        compiled_pattern = self.compile_pattern(pattern)
        matches = []
        
        for match in compiled_pattern.finditer(text):
            matches.append(match.groupdict())
        
        return matches
    
    def find_with_lookahead(self, text: str, pattern: str) -> List[str]:
        """Find matches using positive lookahead"""
        compiled_pattern = self.compile_pattern(pattern)
        return compiled_pattern.findall(text)
    
    def find_with_lookbehind(self, text: str, pattern: str) -> List[str]:
        """Find matches using positive lookbehind"""
        compiled_pattern = self.compile_pattern(pattern)
        return compiled_pattern.findall(text)
    
    def find_repeated_words(self, text: str) -> List[str]:
        """Find repeated words in text"""
        pattern = r'\b(\w+)\s+\1\b'
        compiled_pattern = self.compile_pattern(pattern, re.IGNORECASE)
        return compiled_pattern.findall(text)
    
    def find_palindromes(self, text: str) -> List[str]:
        """Find palindromes in text"""
        words = re.findall(r'\b\w+\b', text)
        palindromes = []
        
        for word in words:
            if word.lower() == word.lower()[::-1] and len(word) > 1:
                palindromes.append(word)
        
        return palindromes
    
    def extract_quoted_text(self, text: str) -> List[str]:
        """Extract text within quotes"""
        pattern = r'"([^"]*)"'
        return self.find_with_lookahead(text, pattern)
    
    def extract_phone_numbers_formatted(self, text: str) -> List[Dict[str, str]]:
        """Extract phone numbers with formatting information"""
        pattern = r'(?P<country_code>\+?1[-.\s]?)?\(?(?P<area_code>[0-9]{3})\)?[-.\s]?(?P<exchange>[0-9]{3})[-.\s]?(?P<number>[0-9]{4})'
        return self.extract_named_groups(text, pattern)

# Exercise 3: Text Validation and Extraction
class TextValidator:
    """Advanced text validation using regex"""
    
    def __init__(self):
        self.validators = {
            'email': self._validate_email,
            'phone': self._validate_phone,
            'url': self._validate_url,
            'credit_card': self._validate_credit_card,
            'ssn': self._validate_ssn,
            'date': self._validate_date,
            'time': self._validate_time,
            'ip_address': self._validate_ip_address,
            'mac_address': self._validate_mac_address
        }
    
    def _validate_email(self, email: str) -> bool:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number"""
        pattern = r'^(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
        return bool(re.match(pattern, phone))
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL"""
        pattern = r'^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$'
        return bool(re.match(pattern, url))
    
    def _validate_credit_card(self, card: str) -> bool:
        """Validate credit card number"""
        pattern = r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
        return bool(re.match(pattern, card))
    
    def _validate_ssn(self, ssn: str) -> bool:
        """Validate SSN"""
        pattern = r'^\d{3}-\d{2}-\d{4}$'
        return bool(re.match(pattern, ssn))
    
    def _validate_date(self, date: str) -> bool:
        """Validate date format"""
        pattern = r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'
        return bool(re.match(pattern, date))
    
    def _validate_time(self, time_str: str) -> bool:
        """Validate time format"""
        pattern = r'^\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?$'
        return bool(re.match(pattern, time_str))
    
    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address"""
        pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        return bool(re.match(pattern, ip))
    
    def _validate_mac_address(self, mac: str) -> bool:
        """Validate MAC address"""
        pattern = r'^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$'
        return bool(re.match(pattern, mac))
    
    def validate(self, text: str, validator_type: str) -> bool:
        """Validate text using specified validator"""
        if validator_type not in self.validators:
            raise ValueError(f"Unknown validator: {validator_type}")
        
        return self.validators[validator_type](text)
    
    def validate_all(self, text: str) -> Dict[str, bool]:
        """Validate text against all validators"""
        results = {}
        for validator_type in self.validators:
            results[validator_type] = self.validate(text, validator_type)
        return results

# Exercise 4: Data Extraction and Parsing
class DataExtractor:
    """Extract structured data from text using regex"""
    
    def __init__(self):
        self.extractors = {
            'emails': self._extract_emails,
            'phones': self._extract_phones,
            'urls': self._extract_urls,
            'dates': self._extract_dates,
            'times': self._extract_times,
            'ip_addresses': self._extract_ip_addresses,
            'mac_addresses': self._extract_mac_addresses,
            'hashtags': self._extract_hashtags,
            'mentions': self._extract_mentions,
            'html_tags': self._extract_html_tags,
            'currency': self._extract_currency,
            'numbers': self._extract_numbers,
            'words': self._extract_words
        }
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers"""
        pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        matches = re.findall(pattern, text)
        return [''.join(match) for match in matches]
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates"""
        pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        return re.findall(pattern, text)
    
    def _extract_times(self, text: str) -> List[str]:
        """Extract times"""
        pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b'
        return re.findall(pattern, text)
    
    def _extract_ip_addresses(self, text: str) -> List[str]:
        """Extract IP addresses"""
        pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        return re.findall(pattern, text)
    
    def _extract_mac_addresses(self, text: str) -> List[str]:
        """Extract MAC addresses"""
        pattern = r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
        return re.findall(pattern, text)
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags"""
        pattern = r'#\w+'
        return re.findall(pattern, text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions"""
        pattern = r'@\w+'
        return re.findall(pattern, text)
    
    def _extract_html_tags(self, text: str) -> List[str]:
        """Extract HTML tags"""
        pattern = r'<[^>]+>'
        return re.findall(pattern, text)
    
    def _extract_currency(self, text: str) -> List[str]:
        """Extract currency amounts"""
        pattern = r'\$[\d,]+\.?\d*'
        return re.findall(pattern, text)
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers"""
        pattern = r'\b\d+(?:\.\d+)?\b'
        return re.findall(pattern, text)
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words"""
        pattern = r'\b\w+\b'
        return re.findall(pattern, text)
    
    def extract(self, text: str, extractor_type: str) -> List[str]:
        """Extract data using specified extractor"""
        if extractor_type not in self.extractors:
            raise ValueError(f"Unknown extractor: {extractor_type}")
        
        return self.extractors[extractor_type](text)
    
    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """Extract all types of data from text"""
        results = {}
        for extractor_type in self.extractors:
            results[extractor_type] = self.extract(text, extractor_type)
        return results

# Exercise 5: Text Manipulation and Replacement
class TextManipulator:
    """Advanced text manipulation using regex"""
    
    def __init__(self):
        self.manipulators = {
            'normalize_whitespace': self._normalize_whitespace,
            'remove_punctuation': self._remove_punctuation,
            'remove_html_tags': self._remove_html_tags,
            'mask_sensitive_data': self._mask_sensitive_data,
            'format_phone_numbers': self._format_phone_numbers,
            'format_dates': self._format_dates,
            'capitalize_words': self._capitalize_words,
            'remove_extra_spaces': self._remove_extra_spaces
        }
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return re.sub(r'[^\w\s]', '', text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        return re.sub(r'<[^>]+>', '', text)
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        # Mask credit card numbers
        text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '****-****-****-****', text)
        
        # Mask SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****', text)
        
        # Mask email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '***@***.***', text)
        
        return text
    
    def _format_phone_numbers(self, text: str, format_style: str = 'standard') -> str:
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
    
    def _format_dates(self, text: str, format_style: str = 'standard') -> str:
        """Format dates in text"""
        date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
        
        def format_date(match):
            month, day, year = match.groups()
            
            if format_style == 'standard':
                return f"{month}/{day}/{year}"
            elif format_style == 'iso':
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                return f"{month}-{day}-{year}"
        
        return re.sub(date_pattern, format_date, text)
    
    def _capitalize_words(self, text: str) -> str:
        """Capitalize first letter of each word"""
        return re.sub(r'\b\w', lambda m: m.group().upper(), text)
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces from text"""
        return re.sub(r' +', ' ', text).strip()
    
    def manipulate(self, text: str, manipulator_type: str, **kwargs) -> str:
        """Manipulate text using specified manipulator"""
        if manipulator_type not in self.manipulators:
            raise ValueError(f"Unknown manipulator: {manipulator_type}")
        
        return self.manipulators[manipulator_type](text, **kwargs)
    
    def replace_words(self, text: str, word_map: Dict[str, str]) -> str:
        """Replace words using dictionary mapping"""
        def replace_func(match):
            word = match.group()
            return word_map.get(word.lower(), word)
        
        return re.sub(r'\b\w+\b', replace_func, text)
    
    def extract_and_replace(self, text: str, pattern: str, replacement_func) -> Tuple[str, List[str]]:
        """Extract matches and replace them using a function"""
        matches = []
        
        def replace_func(match):
            match_text = match.group()
            matches.append(match_text)
            return replacement_func(match_text)
        
        new_text = re.sub(pattern, replace_func, text)
        return new_text, matches

# Exercise 6: Performance Optimization
class OptimizedRegex:
    """Optimized regex operations with caching and profiling"""
    
    def __init__(self):
        self._compiled_patterns = {}
        self._cache = {}
    
    def get_compiled_pattern(self, pattern: str, flags: int = 0) -> Pattern:
        """Get compiled pattern with caching"""
        key = (pattern, flags)
        if key not in self._compiled_patterns:
            self._compiled_patterns[key] = re.compile(pattern, flags)
        return self._compiled_patterns[key]
    
    @functools.lru_cache(maxsize=128)
    def cached_search(self, pattern: str, text: str, flags: int = 0) -> List[str]:
        """Cached regex search"""
        compiled_pattern = self.get_compiled_pattern(pattern, flags)
        return compiled_pattern.findall(text)
    
    def batch_process(self, patterns: List[str], text: str) -> Dict[str, List[str]]:
        """Process multiple patterns efficiently"""
        results = {}
        for pattern in patterns:
            compiled_pattern = self.get_compiled_pattern(pattern)
            results[pattern] = compiled_pattern.findall(text)
        return results
    
    def profile_pattern(self, pattern: str, text: str, iterations: int = 1000) -> Dict[str, Any]:
        """Profile regex pattern performance"""
        compiled_pattern = self.get_compiled_pattern(pattern)
        
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
    
    def compare_patterns(self, patterns: List[str], text: str, iterations: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Compare performance of multiple patterns"""
        results = {}
        
        for pattern in patterns:
            results[pattern] = self.profile_pattern(pattern, text, iterations)
        
        return results

# Exercise 7: Regex Testing and Debugging
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
    
    def run_tests(self) -> Dict[str, Any]:
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
    
    def debug_pattern(self, pattern: str, text: str) -> Dict[str, Any]:
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

# Exercise 8: Real-World Applications
class LogProcessor:
    """Process log files using regex"""
    
    def __init__(self):
        self.log_patterns = {
            'apache': r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)',
            'nginx': r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)"',
            'syslog': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}) (\w+) (.*)',
        }
    
    def parse_apache_log(self, log_line: str) -> Dict[str, Any]:
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
    
    def extract_error_codes(self, log_file: str) -> Dict[str, int]:
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
    
    def find_suspicious_activity(self, log_file: str) -> List[Dict[str, Any]]:
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

# Test Functions
def test_exercises():
    """Test all exercises"""
    print("Testing Module 10 Exercises...")
    
    # Test 1: Basic Pattern Matcher
    print("\n1. Testing Basic Pattern Matcher:")
    basic_matcher = BasicPatternMatcher()
    
    text = "Contact us at support@company.com or call (555) 123-4567. Visit https://example.com"
    
    emails = basic_matcher.find_pattern(text, 'email')
    print(f"Emails found: {emails}")
    
    phones = basic_matcher.find_pattern(text, 'phone')
    print(f"Phone numbers found: {phones}")
    
    urls = basic_matcher.find_pattern(text, 'url')
    print(f"URLs found: {urls}")
    
    # Test 2: Advanced Pattern Matcher
    print("\n2. Testing Advanced Pattern Matcher:")
    advanced_matcher = AdvancedPatternMatcher()
    
    text_with_groups = "John Doe, Jane Smith, Bob Johnson"
    groups = advanced_matcher.extract_groups(text_with_groups, r'(\w+)\s+(\w+)')
    print(f"Name groups: {groups}")
    
    repeated_text = "hello hello world world"
    repeated_words = advanced_matcher.find_repeated_words(repeated_text)
    print(f"Repeated words: {repeated_words}")
    
    # Test 3: Text Validator
    print("\n3. Testing Text Validator:")
    validator = TextValidator()
    
    test_email = "test@example.com"
    is_valid_email = validator.validate(test_email, 'email')
    print(f"Email '{test_email}' is valid: {is_valid_email}")
    
    test_phone = "(555) 123-4567"
    is_valid_phone = validator.validate(test_phone, 'phone')
    print(f"Phone '{test_phone}' is valid: {is_valid_phone}")
    
    # Test 4: Data Extractor
    print("\n4. Testing Data Extractor:")
    extractor = DataExtractor()
    
    test_text = "Email: test@example.com, Phone: (555) 123-4567, URL: https://example.com"
    extracted_data = extractor.extract_all(test_text)
    print(f"Extracted data: {extracted_data}")
    
    # Test 5: Text Manipulator
    print("\n5. Testing Text Manipulator:")
    manipulator = TextManipulator()
    
    test_text = "  Hello   world!  This is a test.  "
    normalized = manipulator.manipulate(test_text, 'normalize_whitespace')
    print(f"Normalized text: '{normalized}'")
    
    masked_text = "My email is test@example.com and my SSN is 123-45-6789"
    masked = manipulator.manipulate(masked_text, 'mask_sensitive_data')
    print(f"Masked text: {masked}")
    
    # Test 6: Optimized Regex
    print("\n6. Testing Optimized Regex:")
    optimized = OptimizedRegex()
    
    test_pattern = r'\b\w+\b'
    test_text = "This is a test text with multiple words"
    
    # Profile the pattern
    profile_result = optimized.profile_pattern(test_pattern, test_text, 1000)
    print(f"Pattern performance: {profile_result['avg_time_per_iteration']:.6f} seconds per iteration")
    
    # Test 7: Regex Tester
    print("\n7. Testing Regex Tester:")
    tester = RegexTester()
    
    # Add test cases
    tester.add_test_case(r'\d+', "123", True, "Should match numbers")
    tester.add_test_case(r'\d+', "abc", False, "Should not match letters")
    tester.add_test_case(r'[a-z]+', "hello", True, "Should match lowercase letters")
    
    # Run tests
    test_results = tester.run_tests()
    print(f"Test results: {test_results['passed']}/{test_results['total']} passed")
    
    # Test 8: Log Processor
    print("\n8. Testing Log Processor:")
    log_processor = LogProcessor()
    
    # Create sample log line
    sample_log = '192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234'
    parsed_log = log_processor.parse_apache_log(sample_log)
    print(f"Parsed log: {parsed_log}")
    
    print("\nAll exercises completed!")

if __name__ == "__main__":
    test_exercises()
