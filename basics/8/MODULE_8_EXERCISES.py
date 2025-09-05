"""
Module 8: String Manipulation and Text Processing - Exercises
Complete these exercises to master string manipulation and text processing.
"""

import re
import string
import unicodedata
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import math

# Exercise 1: Advanced String Operations
class StringProcessor:
    """Advanced string processing utilities"""
    
    def __init__(self):
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()
    
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
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text"""
        pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        matches = re.findall(pattern, text)
        return [''.join(match) for match in matches]
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    def word_frequency(self, text: str) -> Dict[str, int]:
        """Calculate word frequency in text"""
        # Clean and tokenize text
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        words = text.split()
        
        # Count word frequencies
        frequency = Counter(words)
        return dict(frequency)
    
    def n_grams(self, text: str, n: int) -> List[str]:
        """Generate n-grams from text"""
        words = text.split()
        n_grams = []
        for i in range(len(words) - n + 1):
            n_grams.append(' '.join(words[i:i + n]))
        return n_grams

# Exercise 2: Regular Expression Patterns
class RegexPatterns:
    """Collection of useful regular expression patterns"""
    
    def __init__(self):
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b'),
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+'),
            'html_tag': re.compile(r'<[^>]+>'),
            'whitespace': re.compile(r'\s+'),
            'word_boundary': re.compile(r'\b\w+\b'),
            'sentence': re.compile(r'[.!?]+'),
            'paragraph': re.compile(r'\n\s*\n')
        }
    
    def find_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Find all matches for a specific pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return self.patterns[pattern_name].findall(text)
    
    def replace_pattern(self, text: str, pattern_name: str, replacement: str) -> str:
        """Replace all matches of a pattern with replacement"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return self.patterns[pattern_name].sub(replacement, text)
    
    def split_by_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Split text by a pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        return self.patterns[pattern_name].split(text)

# Exercise 3: Text Similarity and Distance
class TextSimilarity:
    """Text similarity and distance calculations"""
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
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
    
    def jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between two strings"""
        set1 = set(s1.split())
        set2 = set(s2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) > 0 else 0
    
    def cosine_similarity(self, s1: str, s2: str) -> float:
        """Calculate cosine similarity between two strings"""
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
    
    def hamming_distance(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between two strings of equal length"""
        if len(s1) != len(s2):
            raise ValueError("Strings must be of equal length")
        
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity between two strings"""
        def jaro_similarity(s1: str, s2: str) -> float:
            if s1 == s2:
                return 1.0
            
            len1, len2 = len(s1), len(s2)
            match_window = max(len1, len2) // 2 - 1
            
            if match_window < 0:
                match_window = 0
            
            s1_matches = [False] * len1
            s2_matches = [False] * len2
            
            matches = 0
            transpositions = 0
            
            # Find matches
            for i in range(len1):
                start = max(0, i - match_window)
                end = min(i + match_window + 1, len2)
                
                for j in range(start, end):
                    if s2_matches[j] or s1[i] != s2[j]:
                        continue
                    s1_matches[i] = s2_matches[j] = True
                    matches += 1
                    break
            
            if matches == 0:
                return 0.0
            
            # Count transpositions
            k = 0
            for i in range(len1):
                if not s1_matches[i]:
                    continue
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
            
            return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        
        jaro = jaro_similarity(s1, s2)
        
        if jaro < 0.7:
            return jaro
        
        # Calculate Winkler prefix bonus
        prefix = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + (0.1 * prefix * (1 - jaro))

# Exercise 4: Text Analysis and NLP
class TextAnalyzer:
    """Advanced text analysis and NLP utilities"""
    
    def __init__(self):
        self.processor = StringProcessor()
        self.similarity = TextSimilarity()
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis using word lists"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'perfect', 'love', 'like',
            'happy', 'joy', 'pleasure', 'delight', 'satisfied', 'content',
            'beautiful', 'gorgeous', 'stunning', 'magnificent', 'superb'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
            'upset', 'worried', 'concerned', 'fear', 'scared', 'anxious',
            'ugly', 'hideous', 'disgusting', 'repulsive', 'atrocious'
        }
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'neither', 'nor'}
        
        words = text.lower().split()
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in negation_words:
                is_negated = True
            
            if word in positive_words:
                if is_negated:
                    negative_count += 1
                else:
                    positive_count += 1
            elif word in negative_words:
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
    
    def text_summarization(self, text: str, num_sentences: int = 3) -> str:
        """Simple text summarization using frequency analysis"""
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies
        word_frequencies = self.processor.word_frequency(text)
        
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
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentence_score(self, sentence: str, word_frequencies: Dict[str, int]) -> float:
        """Calculate score for a sentence"""
        words = sentence.lower().split()
        words = [word.strip('.,!?;:"()[]{}') for word in words]
        words = [word for word in words if word not in self.processor.stop_words and len(word) > 1]
        
        if not words:
            return 0
        
        score = sum(word_frequencies.get(word, 0) for word in words)
        return score / len(words)
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, int]]:
        """Extract keywords from text based on frequency"""
        word_frequencies = self.processor.word_frequency(text)
        
        # Filter out stop words and short words
        filtered_frequencies = {
            word: freq for word, freq in word_frequencies.items()
            if word not in self.processor.stop_words and len(word) > 2
        }
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(filtered_frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:num_keywords]
    
    def text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Character statistics
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Word statistics
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        
        # Sentence statistics
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability metrics (simplified)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'unique_word_count': unique_words,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0
        }

# Exercise 5: Text Encoding and Internationalization
class TextEncoder:
    """Text encoding and internationalization utilities"""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'ascii', 'latin-1', 'cp1252', 'utf-16', 'utf-32']
    
    def detect_encoding(self, text: str) -> str:
        """Detect text encoding (simplified)"""
        try:
            text.encode('ascii')
            return 'ascii'
        except UnicodeEncodeError:
            try:
                text.encode('latin-1')
                return 'latin-1'
            except UnicodeEncodeError:
                return 'utf-8'
    
    def normalize_text(self, text: str, form: str = 'NFC') -> str:
        """Normalize unicode text"""
        return unicodedata.normalize(form, text)
    
    def remove_accents(self, text: str) -> str:
        """Remove accents from text"""
        # Normalize to NFD (decomposed form)
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text
    
    def transliterate(self, text: str) -> str:
        """Transliterate text to ASCII"""
        # Simple transliteration table
        transliteration_table = {
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o', 'ø': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ý': 'y', 'ÿ': 'y',
            'ñ': 'n', 'ç': 'c',
            'Á': 'A', 'À': 'A', 'Â': 'A', 'Ä': 'A', 'Ã': 'A', 'Å': 'A',
            'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
            'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
            'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Ö': 'O', 'Õ': 'O', 'Ø': 'O',
            'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
            'Ý': 'Y', 'Ÿ': 'Y',
            'Ñ': 'N', 'Ç': 'C'
        }
        
        for original, replacement in transliteration_table.items():
            text = text.replace(original, replacement)
        
        return text

# Exercise 6: Performance Testing
class TextPerformanceTester:
    """Performance testing for text processing operations"""
    
    def __init__(self):
        self.processor = StringProcessor()
        self.patterns = RegexPatterns()
        self.analyzer = TextAnalyzer()
    
    def benchmark_operations(self, text: str, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark various text processing operations"""
        results = {}
        
        # Clean text
        start_time = time.time()
        for _ in range(iterations):
            self.processor.clean_text(text)
        results['clean_text'] = (time.time() - start_time) / iterations
        
        # Remove punctuation
        start_time = time.time()
        for _ in range(iterations):
            self.processor.remove_punctuation(text)
        results['remove_punctuation'] = (time.time() - start_time) / iterations
        
        # Extract emails
        start_time = time.time()
        for _ in range(iterations):
            self.processor.extract_emails(text)
        results['extract_emails'] = (time.time() - start_time) / iterations
        
        # Word frequency
        start_time = time.time()
        for _ in range(iterations):
            self.processor.word_frequency(text)
        results['word_frequency'] = (time.time() - start_time) / iterations
        
        # Sentiment analysis
        start_time = time.time()
        for _ in range(iterations):
            self.analyzer.sentiment_analysis(text)
        results['sentiment_analysis'] = (time.time() - start_time) / iterations
        
        return results

# Test Functions
def test_exercises():
    """Test all exercises"""
    print("Testing Module 8 Exercises...")
    
    # Test 1: String Processor
    print("\n1. Testing String Processor:")
    processor = StringProcessor()
    text = "Hello, World! This is a test email: test@example.com and phone: (555) 123-4567"
    
    print(f"Original text: {text}")
    print(f"Cleaned text: {processor.clean_text(text)}")
    print(f"Without punctuation: {processor.remove_punctuation(text)}")
    print(f"Emails found: {processor.extract_emails(text)}")
    print(f"Phone numbers found: {processor.extract_phone_numbers(text)}")
    
    # Test 2: Regex Patterns
    print("\n2. Testing Regex Patterns:")
    patterns = RegexPatterns()
    text = "Contact us at support@company.com or call (555) 123-4567. Visit https://example.com"
    
    print(f"Emails: {patterns.find_pattern(text, 'email')}")
    print(f"URLs: {patterns.find_pattern(text, 'url')}")
    print(f"Phone numbers: {patterns.find_pattern(text, 'phone')}")
    
    # Test 3: Text Similarity
    print("\n3. Testing Text Similarity:")
    similarity = TextSimilarity()
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A quick brown fox jumps over a lazy dog"
    
    print(f"Levenshtein distance: {similarity.levenshtein_distance(text1, text2)}")
    print(f"Jaccard similarity: {similarity.jaccard_similarity(text1, text2):.3f}")
    print(f"Cosine similarity: {similarity.cosine_similarity(text1, text2):.3f}")
    
    # Test 4: Text Analysis
    print("\n4. Testing Text Analysis:")
    analyzer = TextAnalyzer()
    text = "I love this product! It's amazing and wonderful. The quality is outstanding."
    
    sentiment = analyzer.sentiment_analysis(text)
    print(f"Sentiment: {sentiment['sentiment']}")
    print(f"Score: {sentiment['score']:.3f}")
    print(f"Confidence: {sentiment['confidence']:.3f}")
    
    keywords = analyzer.extract_keywords(text, 5)
    print(f"Top keywords: {keywords}")
    
    # Test 5: Text Encoding
    print("\n5. Testing Text Encoding:")
    encoder = TextEncoder()
    text = "Café naïve résumé"
    
    print(f"Original: {text}")
    print(f"Without accents: {encoder.remove_accents(text)}")
    print(f"Transliterated: {encoder.transliterate(text)}")
    
    # Test 6: Performance Testing
    print("\n6. Testing Performance:")
    tester = TextPerformanceTester()
    test_text = "This is a test text with some content. It contains various words and punctuation marks."
    
    results = tester.benchmark_operations(test_text, 100)
    print("Performance results (seconds per operation):")
    for operation, time_taken in results.items():
        print(f"{operation}: {time_taken:.6f}")
    
    print("\nAll exercises completed!")

if __name__ == "__main__":
    test_exercises()
