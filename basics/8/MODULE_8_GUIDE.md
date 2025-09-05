# Module 8: String Manipulation and Text Processing - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master advanced string manipulation techniques in Python
- Implement efficient text processing algorithms
- Work with regular expressions for pattern matching
- Handle text encoding and internationalization
- Build text analysis and NLP preprocessing tools
- Optimize string operations for performance
- Create robust text processing pipelines

## Core Concepts

### 1. Advanced String Operations

#### String Formatting and Templates
```python
# f-strings (Python 3.6+)
name = "Alice"
age = 30
print(f"Hello, {name}! You are {age} years old.")
print(f"Next year you'll be {age + 1}")

# Advanced f-string formatting
pi = 3.14159
print(f"Pi to 2 decimal places: {pi:.2f}")
print(f"Pi in scientific notation: {pi:.2e}")
print(f"Pi with padding: {pi:10.2f}")

# String templates
from string import Template
template = Template("Hello, $name! You are $age years old.")
result = template.substitute(name="Bob", age=25)
print(result)

# Custom template with safe substitution
class SafeTemplate(Template):
    delimiter = '%'
    idpattern = r'[a-z][_a-z0-9]*'

safe_template = SafeTemplate("Hello, %name! You are %age years old.")
result = safe_template.safe_substitute(name="Charlie", age=35)
print(result)
```

#### String Methods and Operations
```python
# String case operations
text = "Hello World"
print(text.upper())        # HELLO WORLD
print(text.lower())        # hello world
print(text.title())        # Hello World
print(text.capitalize())   # Hello world
print(text.swapcase())     # hELLO wORLD

# String validation
text = "Hello123"
print(text.isalpha())      # False (contains numbers)
print(text.isalnum())      # True (alphanumeric)
print(text.isdigit())      # False
print(text.islower())      # False
print(text.isupper())      # False
print(text.isspace())      # False

# String searching and counting
text = "hello world hello python"
print(text.count("hello"))  # 2
print(text.find("world"))   # 6
print(text.find("python"))  # 18
print(text.find("java"))    # -1 (not found)
print(text.index("world"))  # 6
print(text.rfind("hello"))  # 12 (last occurrence)

# String splitting and joining
text = "apple,banana,cherry"
fruits = text.split(",")
print(fruits)  # ['apple', 'banana', 'cherry']

# Split with max splits
text = "a,b,c,d,e"
parts = text.split(",", 2)
print(parts)  # ['a', 'b', 'c,d,e']

# Join with custom separator
fruits = ["apple", "banana", "cherry"]
result = " and ".join(fruits)
print(result)  # apple and banana and cherry

# String stripping and cleaning
text = "  hello world  "
print(text.strip())        # "hello world"
print(text.lstrip())       # "hello world  "
print(text.rstrip())       # "  hello world"

# Strip specific characters
text = "!!!hello world!!!"
print(text.strip("!"))     # "hello world"
```

### 2. Regular Expressions

#### Basic Pattern Matching
```python
import re

# Basic pattern matching
text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"
match = re.search(pattern, text)
if match:
    print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")

# Case-insensitive matching
pattern = re.compile(r"FOX", re.IGNORECASE)
match = pattern.search(text)
if match:
    print(f"Found '{match.group()}' (case-insensitive)")

# Multiple matches
pattern = r"o"
matches = re.findall(pattern, text)
print(f"Found 'o' {len(matches)} times: {matches}")

# Find all matches with positions
for match in re.finditer(r"o", text):
    print(f"Found 'o' at position {match.start()}")
```

#### Character Classes and Quantifiers
```python
# Character classes
text = "The quick brown fox jumps over the lazy dog"
pattern = r"[aeiou]"  # Match any vowel
vowels = re.findall(pattern, text)
print(f"Vowels found: {vowels}")

# Negated character class
pattern = r"[^aeiou\s]"  # Match any non-vowel, non-whitespace
consonants = re.findall(pattern, text)
print(f"Consonants found: {consonants}")

# Predefined character classes
text = "My phone number is 123-456-7890"
pattern = r"\d"  # Match any digit
digits = re.findall(pattern, text)
print(f"Digits found: {digits}")

pattern = r"\w"  # Match any word character
word_chars = re.findall(pattern, text)
print(f"Word characters found: {word_chars}")

# Quantifiers
text = "aaabbbccc"
pattern = r"a*"  # Zero or more 'a's
matches = re.findall(pattern, text)
print(f"Zero or more 'a's: {matches}")

pattern = r"a+"  # One or more 'a's
matches = re.findall(pattern, text)
print(f"One or more 'a's: {matches}")

pattern = r"a{3}"  # Exactly 3 'a's
matches = re.findall(pattern, text)
print(f"Exactly 3 'a's: {matches}")

pattern = r"a{2,4}"  # 2 to 4 'a's
matches = re.findall(pattern, text)
print(f"2 to 4 'a's: {matches}")
```

#### Groups and Capturing
```python
# Simple groups
text = "John Doe, Jane Smith, Bob Johnson"
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
text = "color colour"
pattern = r"colou(?:r)?"  # Non-capturing group
matches = re.findall(pattern, text)
print(f"Color matches (non-capturing): {matches}")

# Backreferences
text = "hello hello world world"
pattern = r"(\w+)\s+\1"  # Match repeated words
matches = re.findall(pattern, text)
print(f"Repeated words: {matches}")
```

#### Substitution and Replacement
```python
# Simple substitution
text = "Hello, World! Hello, Python!"
pattern = r"Hello"
replacement = "Hi"
new_text = re.sub(pattern, replacement, text)
print(f"Substitution: {new_text}")

# Substitution with groups
text = "John Doe, Jane Smith"
pattern = r"(\w+)\s+(\w+)"
replacement = r"\2, \1"  # Last name, First name
new_text = re.sub(pattern, replacement, text)
print(f"Name format change: {new_text}")

# Substitution with function
def replace_numbers(match):
    return str(int(match.group()) * 2)

text = "I have 5 apples and 3 oranges"
pattern = r"\d+"
new_text = re.sub(pattern, replace_numbers, text)
print(f"Doubled numbers: {new_text}")
```

### 3. Text Processing and Analysis

#### Text Cleaning and Preprocessing
```python
import string
import unicodedata

class TextProcessor:
    """Advanced text processing class"""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.punctuation_table = str.maketrans('', '', string.punctuation)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase
        text = text.lower()
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(self.punctuation_table)
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode text"""
        # Normalize to NFD (decomposed form)
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text
    
    def extract_emails(self, text: str) -> list:
        """Extract email addresses from text"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def extract_phone_numbers(self, text: str) -> list:
        """Extract phone numbers from text"""
        pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        matches = re.findall(pattern, text)
        return [''.join(match) for match in matches]
    
    def extract_urls(self, text: str) -> list:
        """Extract URLs from text"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    def word_frequency(self, text: str) -> dict:
        """Calculate word frequency in text"""
        # Clean and tokenize text
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        words = text.split()
        
        # Count word frequencies
        frequency = {}
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
        
        return frequency
    
    def n_grams(self, text: str, n: int) -> list:
        """Generate n-grams from text"""
        words = text.split()
        n_grams = []
        for i in range(len(words) - n + 1):
            n_grams.append(' '.join(words[i:i + n]))
        return n_grams

# Usage example
processor = TextProcessor()
text = "Hello, World! This is a test email: test@example.com and phone: (555) 123-4567"

print("Original text:", text)
print("Cleaned text:", processor.clean_text(text))
print("Without punctuation:", processor.remove_punctuation(text))
print("Emails found:", processor.extract_emails(text))
print("Phone numbers found:", processor.extract_phone_numbers(text))
```

#### Text Similarity and Distance
```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity between two strings"""
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0

def cosine_similarity(s1: str, s2: str) -> float:
    """Calculate cosine similarity between two strings"""
    from collections import Counter
    import math
    
    # Convert to word vectors
    words1 = Counter(s1.split())
    words2 = Counter(s2.split())
    
    # Get all unique words
    all_words = set(words1.keys()).union(set(words2.keys()))
    
    # Create vectors
    vector1 = [words1.get(word, 0) for word in all_words]
    vector2 = [words2.get(word, 0) for word in all_words]
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

# Usage example
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown fox jumps over a lazy dog"

print(f"Levenshtein distance: {levenshtein_distance(text1, text2)}")
print(f"Jaccard similarity: {jaccard_similarity(text1, text2):.3f}")
print(f"Cosine similarity: {cosine_similarity(text1, text2):.3f}")
```

### 4. Text Encoding and Internationalization

#### Unicode and Encoding
```python
import unicodedata
import codecs

def handle_encoding(text: str, encoding: str = 'utf-8') -> bytes:
    """Convert text to bytes with specified encoding"""
    try:
        return text.encode(encoding)
    except UnicodeEncodeError as e:
        print(f"Encoding error: {e}")
        return text.encode(encoding, errors='replace')

def handle_decoding(data: bytes, encoding: str = 'utf-8') -> str:
    """Convert bytes to text with specified encoding"""
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        print(f"Decoding error: {e}")
        return data.decode(encoding, errors='replace')

def normalize_text(text: str, form: str = 'NFC') -> str:
    """Normalize unicode text"""
    return unicodedata.normalize(form, text)

def remove_accents(text: str) -> str:
    """Remove accents from text"""
    # Normalize to NFD (decomposed form)
    text = unicodedata.normalize('NFD', text)
    # Remove combining characters (accents)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

def detect_encoding(file_path: str) -> str:
    """Detect file encoding"""
    import chardet
    
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# Usage example
text = "Café naïve résumé"
print(f"Original: {text}")
print(f"Normalized: {normalize_text(text)}")
print(f"Without accents: {remove_accents(text)}")
```

#### Text Translation and Localization
```python
class TextTranslator:
    """Simple text translator using translation tables"""
    
    def __init__(self):
        self.translation_tables = {
            'leet': str.maketrans('aeiost', '43105t'),
            'reverse': str.maketrans('abcdefghijklmnopqrstuvwxyz', 'zyxwvutsrqponmlkjihgfedcba'),
            'rot13': str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                  'NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm')
        }
    
    def translate(self, text: str, method: str) -> str:
        """Translate text using specified method"""
        if method not in self.translation_tables:
            raise ValueError(f"Unknown translation method: {method}")
        
        return text.translate(self.translation_tables[method])
    
    def create_custom_translation(self, from_chars: str, to_chars: str) -> str:
        """Create custom translation table"""
        return str.maketrans(from_chars, to_chars)

# Usage example
translator = TextTranslator()
text = "Hello World"

print(f"Original: {text}")
print(f"Leet speak: {translator.translate(text, 'leet')}")
print(f"Reverse: {translator.translate(text, 'reverse')}")
print(f"ROT13: {translator.translate(text, 'rot13')}")
```

### 5. Advanced Text Analysis

#### Sentiment Analysis
```python
class SentimentAnalyzer:
    """Simple sentiment analyzer using word lists"""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'perfect', 'love', 'like',
            'happy', 'joy', 'pleasure', 'delight', 'satisfied', 'content'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
            'upset', 'worried', 'concerned', 'fear', 'scared', 'anxious'
        }
        self.negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody'}
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text"""
        words = text.lower().split()
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in self.negation_words:
                is_negated = True
            
            if word in self.positive_words:
                if is_negated:
                    negative_count += 1
                else:
                    positive_count += 1
            elif word in self.negative_words:
                if is_negated:
                    positive_count += 1
                else:
                    negative_count += 1
        
        total_words = len(words)
        if total_words == 0:
            return {'sentiment': 'neutral', 'score': 0, 'confidence': 0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = 'positive'
            score = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = 'negative'
            score = negative_ratio
        else:
            sentiment = 'neutral'
            score = 0
        
        confidence = abs(positive_ratio - negative_ratio)
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count
        }

# Usage example
analyzer = SentimentAnalyzer()
text = "I love this product! It's amazing and wonderful."
result = analyzer.analyze_sentiment(text)
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

#### Text Summarization
```python
class TextSummarizer:
    """Simple text summarizer using frequency analysis"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Summarize text by extracting most important sentences"""
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies
        word_frequencies = self._calculate_word_frequencies(text)
        
        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            score = self._calculate_sentence_score(sentence, word_frequencies)
            sentence_scores.append((score, sentence))
        
        # Sort by score and get top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = [sentence for _, sentence in sentence_scores[:num_sentences]]
        
        # Return in original order
        summary = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary.append(sentence)
        
        return ' '.join(summary)
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_frequencies(self, text: str) -> dict:
        """Calculate word frequencies in text"""
        words = text.lower().split()
        words = [word.strip('.,!?;:"()[]{}') for word in words]
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        frequencies = {}
        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1
        
        return frequencies
    
    def _calculate_sentence_score(self, sentence: str, word_frequencies: dict) -> float:
        """Calculate score for a sentence"""
        words = sentence.lower().split()
        words = [word.strip('.,!?;:"()[]{}') for word in words]
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        if not words:
            return 0
        
        score = sum(word_frequencies.get(word, 0) for word in words)
        return score / len(words)

# Usage example
summarizer = TextSummarizer()
text = """
Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. 
Python has a simple and easy-to-learn syntax. It is widely used in web development, data science, and artificial intelligence. 
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. 
The language has a large standard library and an active community. Python is known for its readability and clean code structure.
"""

summary = summarizer.summarize(text, num_sentences=2)
print("Summary:")
print(summary)
```

## Best Practices

### 1. Performance Optimization
```python
import re
from functools import lru_cache

class OptimizedTextProcessor:
    """Optimized text processor with caching and compiled regex"""
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Pre-compile common patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
    
    @lru_cache(maxsize=128)
    def clean_text(self, text: str) -> str:
        """Clean text with caching"""
        # Use compiled regex for better performance
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def extract_emails_fast(self, text: str) -> list:
        """Fast email extraction using compiled regex"""
        return self.email_pattern.findall(text)
    
    def extract_phone_numbers_fast(self, text: str) -> list:
        """Fast phone number extraction using compiled regex"""
        matches = self.phone_pattern.findall(text)
        return [''.join(match) for match in matches]
    
    def extract_urls_fast(self, text: str) -> list:
        """Fast URL extraction using compiled regex"""
        return self.url_pattern.findall(text)
```

### 2. Error Handling and Validation
```python
class TextValidator:
    """Text validation and error handling"""
    
    def __init__(self):
        self.max_length = 10000
        self.min_length = 1
    
    def validate_text(self, text: str) -> dict:
        """Validate text input"""
        errors = []
        warnings = []
        
        if not isinstance(text, str):
            errors.append("Text must be a string")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        if len(text) < self.min_length:
            errors.append(f"Text must be at least {self.min_length} characters long")
        
        if len(text) > self.max_length:
            errors.append(f"Text must be no more than {self.max_length} characters long")
        
        if not text.strip():
            errors.append("Text cannot be empty or only whitespace")
        
        # Check for suspicious patterns
        if 'script' in text.lower():
            warnings.append("Text contains 'script' - potential security risk")
        
        if text.count('@') > 10:
            warnings.append("Text contains many '@' symbols - might be spam")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def safe_extract_emails(self, text: str) -> list:
        """Safely extract emails with error handling"""
        try:
            validation = self.validate_text(text)
            if not validation['valid']:
                print(f"Text validation failed: {validation['errors']}")
                return []
            
            pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            return pattern.findall(text)
        
        except Exception as e:
            print(f"Error extracting emails: {e}")
            return []
```

## Quick Checks

### Check 1: String Formatting
```python
# What will this print?
name = "Alice"
age = 30
print(f"Hello, {name}! You are {age} years old.")
```

### Check 2: Regular Expressions
```python
# What will this return?
import re
text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"
matches = re.findall(pattern, text)
print(matches)
```

### Check 3: String Methods
```python
# What will this print?
text = "  Hello World  "
print(text.strip().upper())
```

## Lab Problems

### Lab 1: Text Analysis Tool
Build a comprehensive text analysis tool that can extract entities, analyze sentiment, and generate summaries.

### Lab 2: Text Preprocessing Pipeline
Create a robust text preprocessing pipeline for NLP applications with cleaning, normalization, and validation.

### Lab 3: Pattern Matching Engine
Implement a pattern matching engine that can find and replace complex patterns in text using regular expressions.

### Lab 4: Text Similarity System
Build a text similarity system that can compare documents and find similar content using various similarity metrics.

## AI Code Comparison
When working with AI-generated text processing code, evaluate:
- **Pattern accuracy** - are regular expressions correct and comprehensive?
- **Performance considerations** - is the code optimized for large text processing?
- **Error handling** - are edge cases and invalid inputs handled properly?
- **Unicode support** - does the code handle international characters correctly?
- **Security implications** - are there any potential security vulnerabilities in the text processing?

## Next Steps
- Learn about natural language processing libraries like NLTK and spaCy
- Master advanced text analysis techniques and machine learning
- Explore text mining and information extraction
- Study text generation and language models
- Understand text processing in web applications and APIs
