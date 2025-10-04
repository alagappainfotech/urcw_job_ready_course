 MODULE-8
1. Testing String Processor:
Original text: Hello, World! This is a test email: test@example.com and phone: (555) 123-4567
Cleaned text: hello, world! this is a test email: test@example.com and phone: (555) 123-4567
Without punctuation: Hello World This is a test email testexamplecom and phone 555 1234567
Emails found: ['test@example.com']
Phone numbers found: ['5551234567']

2. Testing Regex Patterns:
Emails: ['support@company.com']
URLs: ['https://example.com']
Phone numbers: [('', '555', '123', '4567')]

3. Testing Text Similarity:
Levenshtein distance: 6
Jaccard similarity: 0.636
Cosine similarity: 0.778
4. Testing Text Analysis:
Sentiment: positive
Score: 0.167
Confidence: 0.167
Top keywords: [('love', 1), ('product', 1), ('its', 1), ('amazing', 1), ('wonderful', 1)]

5. Testing Text Encoding:
Original: Café naïve résumé
Without accents: Cafe naive resume
Transliterated: Cafe naive resume

6. Testing Performance:
Performance results (seconds per operation):
clean_text: 0.000007
remove_punctuation: 0.000005
extract_emails: 0.000004
word_frequency: 0.000018
sentiment_analysis: 0.000006

All exercises completed!