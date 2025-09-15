# Module 11: Data Science and Machine Learning - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master data manipulation and analysis with Pandas and NumPy
- Create compelling visualizations using Matplotlib and Seaborn
- Implement machine learning algorithms with Scikit-learn
- Handle real-world datasets and perform data preprocessing
- Build predictive models and evaluate their performance
- Apply statistical analysis and hypothesis testing
- Deploy machine learning models for production use

## Core Concepts

### 1. Data Science Workflow
The data science process typically follows these steps:
1. **Data Collection** - Gathering data from various sources
2. **Data Cleaning** - Handling missing values, outliers, and inconsistencies
3. **Exploratory Data Analysis (EDA)** - Understanding data patterns and relationships
4. **Feature Engineering** - Creating new features from existing data
5. **Model Building** - Training machine learning models
6. **Model Evaluation** - Assessing model performance
7. **Model Deployment** - Making models available for production use

### 2. Essential Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

### 3. Data Manipulation with Pandas
```python
# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['IT', 'HR', 'IT', 'Finance']
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())  # First 5 rows
print(df.info())  # Data types and memory usage
print(df.describe())  # Statistical summary
print(df.shape)  # Dimensions

# Data selection and filtering
young_employees = df[df['Age'] < 30]
it_department = df[df['Department'] == 'IT']
high_salary = df[df['Salary'] > 55000]

# Grouping and aggregation
dept_stats = df.groupby('Department').agg({
    'Age': ['mean', 'std'],
    'Salary': ['mean', 'max', 'min']
})

# Handling missing data
df.isnull().sum()  # Count missing values
df.dropna()  # Remove rows with missing values
df.fillna(df.mean())  # Fill with mean values
df.fillna(method='ffill')  # Forward fill
```

### 4. Data Visualization
```python
# Matplotlib basics
plt.figure(figsize=(10, 6))
plt.plot(df['Age'], df['Salary'], 'o')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.grid(True)
plt.show()

# Seaborn for statistical plots
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Scatter plot with regression line
sns.scatterplot(data=df, x='Age', y='Salary', hue='Department')
sns.regplot(data=df, x='Age', y='Salary', scatter=False)

# Distribution plots
sns.histplot(df['Age'], kde=True)
sns.boxplot(data=df, x='Department', y='Salary')

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### 5. Machine Learning with Scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare data
X = df[['Age']]  # Features
y = df['Salary']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.2f}')
```

## Advanced Topics

### 1. Feature Engineering
```python
# Creating new features
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], 
                        labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
df['Salary_Range'] = pd.cut(df['Salary'], bins=3, labels=['Low', 'Medium', 'High'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Department'])

# Feature scaling
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Min-Max scaling (0-1 range)
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df[['Age', 'Salary']])

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df[['Age', 'Salary']])
```

### 2. Classification Models
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Prepare classification data
X = df[['Age', 'Salary']]
y = df['Department']  # Target for classification

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: {accuracy:.3f}')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')
```

### 3. Clustering and Unsupervised Learning
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering
X = df[['Age', 'Salary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Salary', hue='Cluster')
plt.title('K-Means Clustering')
plt.show()

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.show()
```

### 4. Time Series Analysis
```python
import pandas as pd
from datetime import datetime, timedelta

# Create time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_data = pd.DataFrame({
    'Date': dates,
    'Value': np.random.randn(len(dates)).cumsum() + 100
})

# Set date as index
ts_data.set_index('Date', inplace=True)

# Time series analysis
ts_data['MA_7'] = ts_data['Value'].rolling(window=7).mean()
ts_data['MA_30'] = ts_data['Value'].rolling(window=30).mean()

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['Value'], label='Original')
plt.plot(ts_data.index, ts_data['MA_7'], label='7-day MA')
plt.plot(ts_data.index, ts_data['MA_30'], label='30-day MA')
plt.legend()
plt.title('Time Series with Moving Averages')
plt.show()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_data['Value'], model='additive', period=30)
decomposition.plot()
plt.show()
```

### 5. Model Evaluation and Validation
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve

# Classification metrics
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC for binary classification
if len(np.unique(y_test)) == 2:
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {auc_score:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

## Practical Applications

### 1. Customer Segmentation
```python
# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Select features for segmentation
features = ['Age', 'Annual_Income', 'Spending_Score']
X = customer_data[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Elbow method
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Final clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=customer_data, x='Annual_Income', y='Spending_Score', 
                hue='Cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.show()
```

### 2. Sales Forecasting
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepare time series data for forecasting
def create_features(df, target_col, lags=5):
    """Create lagged features for time series forecasting"""
    for i in range(1, lags + 1):
        df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
    return df

# Create features
ts_data = create_features(ts_data, 'Value', lags=7)

# Remove rows with NaN values
ts_data = ts_data.dropna()

# Prepare features and target
X = ts_data.drop('Value', axis=1)
y = ts_data['Value']

# Split data (use last 30 days for testing)
split_point = len(ts_data) - 30
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'{name}:')
    print(f'  MAE: {mae:.2f}')
    print(f'  MSE: {mse:.2f}')
    print()
```

### 3. Text Analysis and NLP
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re

# Sample text data
text_data = pd.DataFrame({
    'text': [
        'I love this product! It works great.',
        'This is terrible. Waste of money.',
        'Good quality, fast delivery.',
        'Poor customer service.',
        'Excellent value for money.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
})

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

text_data['processed_text'] = text_data['text'].apply(preprocess_text)

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', MultinomialNB())
])

# Train model
X = text_data['processed_text']
y = text_data['sentiment']

text_clf.fit(X, y)

# Test on new text
new_text = "This product is amazing!"
prediction = text_clf.predict([new_text])
print(f"Prediction: {prediction[0]}")
```

## Best Practices

### 1. Data Quality and Preprocessing
```python
def data_quality_report(df):
    """Generate comprehensive data quality report"""
    print("Data Quality Report")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nDuplicate rows:")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("\nStatistical summary:")
    print(df.describe())

# Handle outliers
def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
```

### 2. Model Selection and Validation
```python
def compare_models(X, y, models):
    """Compare multiple models using cross-validation"""
    results = {}
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
    
    # Display results
    for name, result in results.items():
        print(f"{name}: {result['mean_score']:.3f} (+/- {result['std_score'] * 2:.3f})")
    
    return results

# Usage
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = compare_models(X, y, models)
```

### 3. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, k=10):
    """Select best features using multiple methods"""
    
    # Method 1: Univariate selection
    selector_univariate = SelectKBest(score_func=f_classif, k=k)
    X_selected_uni = selector_univariate.fit_transform(X, y)
    
    # Method 2: Recursive Feature Elimination
    rf = RandomForestClassifier(random_state=42)
    selector_rfe = RFE(rf, n_features_to_select=k)
    X_selected_rfe = selector_rfe.fit_transform(X, y)
    
    # Method 3: Feature importance from Random Forest
    rf.fit(X, y)
    feature_importance = rf.feature_importances_
    
    return {
        'univariate': selector_univariate.get_support(),
        'rfe': selector_rfe.get_support(),
        'importance': feature_importance
    }
```

## Quick Checks

### Check 1: Data Manipulation
```python
# What will this return?
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
result = df[df['A'] > 2]['B'].mean()
print(result)
```

### Check 2: Machine Learning
```python
# What will this print?
from sklearn.linear_model import LinearRegression
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[5]])
print(prediction[0])
```

### Check 3: Visualization
```python
# What type of plot will this create?
sns.pairplot(df, hue='category')
```

## Lab Problems

### Lab 1: Customer Churn Prediction
Build a model to predict customer churn using historical data and behavioral features.

### Lab 2: Stock Price Prediction
Create a time series model to predict stock prices using historical data and technical indicators.

### Lab 3: Image Classification
Implement an image classification system using deep learning techniques.

### Lab 4: Recommendation System
Build a collaborative filtering recommendation system for e-commerce.

## AI Code Comparison
When working with AI-generated data science code, evaluate:
- **Data preprocessing** - are missing values and outliers handled appropriately?
- **Feature engineering** - are new features created that improve model performance?
- **Model selection** - are appropriate algorithms chosen for the problem type?
- **Evaluation metrics** - are the right metrics used to assess model performance?
- **Code efficiency** - is the code optimized for large datasets?

## Next Steps
- Learn about deep learning with TensorFlow and PyTorch
- Master advanced statistical analysis and hypothesis testing
- Explore big data processing with Spark and Dask
- Study model deployment and MLOps practices
- Understand ethical considerations in AI and machine learning
