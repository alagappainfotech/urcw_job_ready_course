# Module 15: AI and Machine Learning Integration - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Integrate AI and ML models into Python applications
- Use popular ML libraries like TensorFlow, PyTorch, and Scikit-learn
- Implement natural language processing and computer vision
- Build recommendation systems and predictive analytics
- Deploy ML models in production environments
- Understand ethical considerations in AI development
- Master prompt engineering and AI-assisted development

## Core Concepts

### 1. Machine Learning Pipeline
A complete ML pipeline includes data preparation, model training, evaluation, and deployment.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pickle

class MLPipeline:
    """Complete machine learning pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
    
    def load_data(self, file_path):
        """Load data from file"""
        self.data = pd.read_csv(file_path)
        return self.data
    
    def preprocess_data(self, target_column, feature_columns=None):
        """Preprocess data for training"""
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != target_column]
        
        self.feature_columns = feature_columns
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train machine learning model"""
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            from sklearn.svm import SVC
            self.model = SVC(random_state=42)
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
    
    def predict(self, new_data):
        """Make predictions on new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess new data
        new_data_scaled = self.scaler.transform(new_data[self.feature_columns])
        
        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        return predictions, probabilities

# Usage example
pipeline = MLPipeline()
data = pipeline.load_data('data.csv')
X_train, X_test, y_train, y_test = pipeline.preprocess_data('target')

model = pipeline.train_model(X_train, y_train, 'random_forest')
predictions = pipeline.evaluate_model(X_test, y_test)

pipeline.save_model('model.pkl', 'scaler.pkl')
```

### 2. Natural Language Processing
NLP techniques for text analysis and processing.

```python
import nltk
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

class NLPProcessor:
    """Natural Language Processing processor"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        result = self.sentiment_analyzer(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract keywords from text using TF-IDF"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform text
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:num_keywords]
    
    def cluster_documents(self, documents, num_clusters=5):
        """Cluster documents using K-means"""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Group documents by cluster
        clustered_docs = {}
        for i, cluster in enumerate(clusters):
            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(documents[i])
        
        return clustered_docs
    
    def get_bert_embeddings(self, text):
        """Get BERT embeddings for text"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()

# Usage example
nlp = NLPProcessor()

text = "Apple Inc. is a technology company based in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."

# Extract entities
entities = nlp.extract_entities(text)
print("Entities:", entities)

# Analyze sentiment
sentiment = nlp.analyze_sentiment(text)
print("Sentiment:", sentiment)

# Extract keywords
keywords = nlp.extract_keywords(text)
print("Keywords:", keywords)
```

### 3. Computer Vision
Image processing and analysis using computer vision techniques.

```python
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class ComputerVisionProcessor:
    """Computer Vision processor for image analysis"""
    
    def __init__(self):
        self.vgg_model = VGG16(weights='imagenet', include_top=False)
        self.resnet_model = ResNet50(weights='imagenet', include_top=False)
    
    def load_image(self, image_path):
        """Load image from file"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(self, image, width, height):
        """Resize image to specified dimensions"""
        return cv2.resize(image, (width, height))
    
    def detect_edges(self, image):
        """Detect edges in image using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def detect_faces(self, image):
        """Detect faces in image using Haar cascades"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_colors(self, image, num_colors=10):
        """Extract dominant colors from image"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get color frequencies
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        color_frequencies = color_counts / len(labels)
        
        return colors, color_frequencies
    
    def extract_features(self, image, model_type='vgg16'):
        """Extract features from image using pre-trained model"""
        # Resize image to model input size
        if model_type == 'vgg16':
            target_size = (224, 224)
            model = self.vgg_model
        elif model_type == 'resnet50':
            target_size = (224, 224)
            model = self.resnet_model
        
        # Resize image
        resized_image = cv2.resize(image, target_size)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized_image)
        
        # Convert to array and preprocess
        img_array = image.img_to_array(pil_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_array)
        features = features.flatten()
        
        return features
    
    def classify_image(self, image, model_path):
        """Classify image using trained model"""
        # Load trained model
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess image
        resized_image = cv2.resize(image, (224, 224))
        pil_image = Image.fromarray(resized_image)
        img_array = image.img_to_array(pil_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_class, confidence
    
    def segment_image(self, image, num_segments=5):
        """Segment image using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_segments, random_state=42)
        kmeans.fit(pixels)
        
        # Get cluster labels
        labels = kmeans.labels_
        
        # Reshape labels back to image shape
        segmented_image = labels.reshape(image.shape[:2])
        
        return segmented_image

# Usage example
cv = ComputerVisionProcessor()

# Load image
image = cv.load_image('image.jpg')

# Detect edges
edges = cv.detect_edges(image)

# Detect faces
faces = cv.detect_faces(image)

# Extract colors
colors, frequencies = cv.extract_colors(image)

# Extract features
features = cv.extract_features(image, 'vgg16')
```

### 4. Recommendation Systems
Building recommendation systems using collaborative and content-based filtering.

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class RecommendationSystem:
    """Recommendation system implementation"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.item_features = None
        self.user_features = None
    
    def create_user_item_matrix(self, ratings_df):
        """Create user-item rating matrix"""
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        return self.user_item_matrix
    
    def collaborative_filtering(self, user_id, num_recommendations=10):
        """Collaborative filtering recommendations"""
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created")
        
        # Get user ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate user similarity
        user_similarity = cosine_similarity(
            self.user_item_matrix.loc[user_id].values.reshape(1, -1),
            self.user_item_matrix
        )[0]
        
        # Get similar users
        similar_users = np.argsort(user_similarity)[::-1][1:11]  # Top 10 similar users
        
        # Calculate weighted ratings
        weighted_ratings = np.zeros(self.user_item_matrix.shape[1])
        
        for similar_user in similar_users:
            similarity = user_similarity[similar_user]
            user_ratings_similar = self.user_item_matrix.iloc[similar_user]
            weighted_ratings += similarity * user_ratings_similar.values
        
        # Get items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Get recommendations
        recommendations = []
        for item in unrated_items:
            item_idx = self.user_item_matrix.columns.get_loc(item)
            if weighted_ratings[item_idx] > 0:
                recommendations.append((item, weighted_ratings[item_idx]))
        
        # Sort by rating and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def content_based_filtering(self, item_id, num_recommendations=10):
        """Content-based filtering recommendations"""
        if self.item_similarity_matrix is None:
            raise ValueError("Item similarity matrix not created")
        
        # Get item similarities
        item_similarities = self.item_similarity_matrix[item_id]
        
        # Get top similar items
        similar_items = np.argsort(item_similarities)[::-1][1:num_recommendations+1]
        
        recommendations = []
        for similar_item in similar_items:
            similarity = item_similarities[similar_item]
            recommendations.append((similar_item, similarity))
        
        return recommendations
    
    def matrix_factorization(self, num_factors=50, num_iterations=100):
        """Matrix factorization using SVD"""
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created")
        
        # Convert to sparse matrix
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Perform SVD
        svd = TruncatedSVD(n_components=num_factors, random_state=42)
        user_factors = svd.fit_transform(sparse_matrix)
        item_factors = svd.components_.T
        
        # Reconstruct matrix
        reconstructed_matrix = user_factors @ item_factors.T
        
        return user_factors, item_factors, reconstructed_matrix
    
    def hybrid_recommendation(self, user_id, item_id, num_recommendations=10):
        """Hybrid recommendation combining collaborative and content-based filtering"""
        # Collaborative filtering recommendations
        collab_recs = self.collaborative_filtering(user_id, num_recommendations)
        
        # Content-based filtering recommendations
        content_recs = self.content_based_filtering(item_id, num_recommendations)
        
        # Combine recommendations
        hybrid_recs = {}
        
        # Add collaborative filtering scores
        for item, score in collab_recs:
            hybrid_recs[item] = score * 0.6  # Weight: 60%
        
        # Add content-based filtering scores
        for item, score in content_recs:
            if item in hybrid_recs:
                hybrid_recs[item] += score * 0.4  # Weight: 40%
            else:
                hybrid_recs[item] = score * 0.4
        
        # Sort and return top recommendations
        sorted_recs = sorted(hybrid_recs.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:num_recommendations]

# Usage example
rec_system = RecommendationSystem()

# Create sample data
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'item_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5]
}
ratings_df = pd.DataFrame(ratings_data)

# Create user-item matrix
user_item_matrix = rec_system.create_user_item_matrix(ratings_df)

# Get collaborative filtering recommendations
collab_recs = rec_system.collaborative_filtering(1, 5)
print("Collaborative Filtering Recommendations:", collab_recs)
```

### 5. Deep Learning with TensorFlow
Building and training deep learning models.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class DeepLearningModel:
    """Deep learning model builder and trainer"""
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_mlp(self, input_dim, hidden_layers, output_dim, activation='relu'):
        """Build Multi-Layer Perceptron"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        model.add(Dropout(0.2))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(output_dim, activation='softmax'))
        
        self.model = model
        return model
    
    def build_cnn(self, input_shape, num_classes):
        """Build Convolutional Neural Network"""
        model = Sequential()
        
        # Convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        return model
    
    def build_lstm(self, input_shape, hidden_units, output_dim):
        """Build LSTM network"""
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(hidden_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(hidden_units, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Usage example
dl_model = DeepLearningModel()

# Build MLP
model = dl_model.build_mlp(
    input_dim=784,
    hidden_layers=[128, 64, 32],
    output_dim=10
)

# Compile model
dl_model.compile_model()

# Train model (assuming you have training data)
# history = dl_model.train_model(X_train, y_train, X_val, y_val)

# Evaluate model
# loss, accuracy = dl_model.evaluate_model(X_test, y_test)
```

### 6. AI-Assisted Development
Using AI tools for code generation, debugging, and optimization.

```python
import openai
import requests
import json
from typing import List, Dict, Any

class AIAssistant:
    """AI assistant for code generation and debugging"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
    
    def generate_code(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate code using AI"""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    
    def debug_code(self, code: str, error_message: str) -> str:
        """Debug code using AI"""
        prompt = f"""
        Debug the following Python code:
        
        Code:
        {code}
        
        Error:
        {error_message}
        
        Please provide a corrected version and explain what was wrong.
        """
        
        response = self.generate_code(prompt)
        return response
    
    def optimize_code(self, code: str) -> str:
        """Optimize code using AI"""
        prompt = f"""
        Optimize the following Python code for better performance and readability:
        
        {code}
        
        Please provide an optimized version with explanations.
        """
        
        response = self.generate_code(prompt)
        return response
    
    def generate_tests(self, code: str) -> str:
        """Generate unit tests for code"""
        prompt = f"""
        Generate comprehensive unit tests for the following Python code:
        
        {code}
        
        Please include edge cases and error handling.
        """
        
        response = self.generate_code(prompt)
        return response
    
    def explain_code(self, code: str) -> str:
        """Explain code using AI"""
        prompt = f"""
        Explain the following Python code in detail:
        
        {code}
        
        Please explain what each part does and how it works.
        """
        
        response = self.generate_code(prompt)
        return response
    
    def refactor_code(self, code: str, requirements: str) -> str:
        """Refactor code according to requirements"""
        prompt = f"""
        Refactor the following Python code according to these requirements:
        
        Requirements: {requirements}
        
        Code:
        {code}
        
        Please provide the refactored code with explanations.
        """
        
        response = self.generate_code(prompt)
        return response

# Usage example
# ai = AIAssistant("your-api-key")

# Generate code
# code = ai.generate_code("Create a function to calculate fibonacci numbers")

# Debug code
# debugged_code = ai.debug_code("def add(a, b): return a + b", "TypeError: unsupported operand type(s)")

# Optimize code
# optimized_code = ai.optimize_code("def slow_function(): return [i*2 for i in range(1000000)]")

# Generate tests
# tests = ai.generate_tests("def add(a, b): return a + b")

# Explain code
# explanation = ai.explain_code("def quicksort(arr): return sorted(arr)")
```

### 7. Model Deployment
Deploying ML models in production environments.

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from werkzeug.exceptions import BadRequest
import logging

class ModelServer:
    """Model server for deploying ML models"""
    
    def __init__(self, model_path: str, scaler_path: str):
        self.app = Flask(__name__)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.setup_routes()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'model_loaded': self.model is not None})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                # Get input data
                data = request.get_json()
                if not data or 'features' not in data:
                    raise BadRequest('Missing features in request')
                
                features = np.array(data['features']).reshape(1, -1)
                
                # Preprocess features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                prediction = self.model.predict(features_scaled)
                probability = self.model.predict_proba(features_scaled)
                
                # Return prediction
                return jsonify({
                    'prediction': prediction[0].tolist(),
                    'probability': probability[0].tolist(),
                    'status': 'success'
                })
                
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint"""
            try:
                # Get input data
                data = request.get_json()
                if not data or 'features' not in data:
                    raise BadRequest('Missing features in request')
                
                features = np.array(data['features'])
                
                # Preprocess features
                features_scaled = self.scaler.transform(features)
                
                # Make predictions
                predictions = self.model.predict(features_scaled)
                probabilities = self.model.predict_proba(features_scaled)
                
                # Return predictions
                return jsonify({
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'status': 'success'
                })
                
            except Exception as e:
                self.logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the model server"""
        self.logger.info(f"Starting model server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Usage example
# server = ModelServer('model.pkl', 'scaler.pkl')
# server.run(host='0.0.0.0', port=5000)

# Docker deployment
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "model_server.py"]
"""

# Kubernetes deployment
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-server
  template:
    metadata:
      labels:
        app: ml-model-server
    spec:
      containers:
      - name: ml-model-server
        image: ml-model-server:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
"""
```

## Best Practices

### 1. Model Versioning and Management
```python
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

class ModelVersioning:
    """Model versioning and management"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_model(self, model, metrics: Dict[str, float], params: Dict[str, Any], 
                  model_name: str, version: str = None):
        """Log model to MLflow"""
        with mlflow.start_run():
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=model_name
            )
            
            # Set version
            if version:
                mlflow.set_tag("version", version)
    
    def load_model(self, model_name: str, version: str = None):
        """Load model from MLflow"""
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        model = mlflow.sklearn.load_model(model_uri)
        return model
    
    def compare_models(self, model_name: str, versions: List[str]):
        """Compare different model versions"""
        results = {}
        
        for version in versions:
            model = self.load_model(model_name, version)
            # Evaluate model and store results
            # This would depend on your specific evaluation metrics
            results[version] = {"accuracy": 0.95, "precision": 0.94, "recall": 0.93}
        
        return results

# Usage example
# model_versioning = ModelVersioning("my_experiment")
# model_versioning.log_model(model, metrics, params, "my_model", "v1.0")
```

### 2. Ethical AI Considerations
```python
class EthicalAIFramework:
    """Framework for ethical AI development"""
    
    def __init__(self):
        self.bias_metrics = {}
        self.fairness_metrics = {}
    
    def check_bias(self, model, X_test, y_test, sensitive_features):
        """Check for bias in model predictions"""
        predictions = model.predict(X_test)
        
        # Calculate bias metrics for each sensitive feature
        for feature in sensitive_features:
            feature_values = X_test[feature].unique()
            
            for value in feature_values:
                mask = X_test[feature] == value
                group_predictions = predictions[mask]
                group_actual = y_test[mask]
                
                # Calculate accuracy for this group
                accuracy = (group_predictions == group_actual).mean()
                self.bias_metrics[f"{feature}_{value}"] = accuracy
        
        return self.bias_metrics
    
    def check_fairness(self, model, X_test, y_test, sensitive_features):
        """Check for fairness in model predictions"""
        predictions = model.predict(X_test)
        
        # Calculate fairness metrics
        for feature in sensitive_features:
            feature_values = X_test[feature].unique()
            
            # Calculate positive prediction rate for each group
            for value in feature_values:
                mask = X_test[feature] == value
                group_predictions = predictions[mask]
                positive_rate = (group_predictions == 1).mean()
                self.fairness_metrics[f"{feature}_{value}_positive_rate"] = positive_rate
        
        return self.fairness_metrics
    
    def generate_bias_report(self):
        """Generate bias and fairness report"""
        report = {
            "bias_metrics": self.bias_metrics,
            "fairness_metrics": self.fairness_metrics,
            "recommendations": self._generate_recommendations()
        }
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations for addressing bias and fairness issues"""
        recommendations = []
        
        # Check for significant bias
        if self.bias_metrics:
            max_accuracy = max(self.bias_metrics.values())
            min_accuracy = min(self.bias_metrics.values())
            
            if max_accuracy - min_accuracy > 0.1:
                recommendations.append("Significant bias detected. Consider retraining with balanced data.")
        
        # Check for fairness issues
        if self.fairness_metrics:
            positive_rates = [v for k, v in self.fairness_metrics.items() if 'positive_rate' in k]
            if positive_rates:
                max_rate = max(positive_rates)
                min_rate = min(positive_rates)
                
                if max_rate - min_rate > 0.2:
                    recommendations.append("Significant fairness issues detected. Consider using fairness constraints.")
        
        return recommendations

# Usage example
# ethical_ai = EthicalAIFramework()
# bias_metrics = ethical_ai.check_bias(model, X_test, y_test, ['gender', 'race'])
# fairness_metrics = ethical_ai.check_fairness(model, X_test, y_test, ['gender', 'race'])
# report = ethical_ai.generate_bias_report()
```

## Quick Checks

### Check 1: ML Pipeline
```python
# What will this code do?
pipeline = MLPipeline()
data = pipeline.load_data('data.csv')
X_train, X_test, y_train, y_test = pipeline.preprocess_data('target')
model = pipeline.train_model(X_train, y_train)
```

### Check 2: NLP Processing
```python
# What will this return?
nlp = NLPProcessor()
text = "Apple Inc. is a great company!"
entities = nlp.extract_entities(text)
```

### Check 3: Model Deployment
```python
# What will this endpoint return?
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction})
```

## Lab Problems

### Lab 1: End-to-End ML Pipeline
Build a complete machine learning pipeline from data collection to model deployment.

### Lab 2: AI-Powered Application
Create an AI-powered application that combines multiple ML models and provides intelligent recommendations.

### Lab 3: Ethical AI System
Implement an ethical AI system with bias detection, fairness monitoring, and explainable AI features.

### Lab 4: Production ML Platform
Build a production-ready ML platform with model versioning, monitoring, and automated retraining.

## AI Code Comparison
When working with AI-generated ML code, evaluate:
- **Model performance** - does the model achieve good accuracy and generalization?
- **Code efficiency** - is the code optimized for performance and memory usage?
- **Error handling** - are edge cases and errors properly handled?
- **Scalability** - can the code handle large datasets and high traffic?
- **Ethical considerations** - are bias and fairness issues addressed?

## Next Steps
- Learn about advanced ML techniques like deep learning and reinforcement learning
- Master MLOps and model deployment strategies
- Explore AI ethics and responsible AI development
- Study AI in specific domains like healthcare, finance, and education
- Understand the future of AI and emerging technologies
