# Module 9: Web Development with Flask - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Build dynamic web applications using Flask
- Implement RESTful APIs and web services
- Handle HTTP requests, responses, and routing
- Work with templates, forms, and user sessions
- Integrate databases with Flask applications
- Deploy Flask applications to production
- Apply security best practices in web development

## Core Concepts

### 1. Flask Framework Overview
Flask is a lightweight WSGI web application framework that provides:
- **Minimal core** with extensibility through extensions
- **Jinja2 templating** engine for dynamic HTML generation
- **Werkzeug WSGI** toolkit for web server interface
- **Built-in development server** for testing
- **RESTful request handling** with decorators

### 2. Basic Flask Application Structure
```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({'status': 'success', 'data': data})
    return jsonify({'message': 'Hello from API'})

if __name__ == '__main__':
    app.run(debug=True)
```

### 3. Routing and URL Patterns
```python
# Basic routes
@app.route('/')
def index():
    return 'Home Page'

@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post ID: {post_id}'

# HTTP methods
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login
        pass
    return render_template('login.html')

# URL building
from flask import url_for
url = url_for('show_user', username='john')
```

### 4. Request Handling
```python
from flask import request

@app.route('/api/users', methods=['POST'])
def create_user():
    # Get JSON data
    data = request.get_json()
    
    # Get form data
    username = request.form.get('username')
    email = request.form.get('email')
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    
    # Get headers
    user_agent = request.headers.get('User-Agent')
    
    return jsonify({'status': 'success'})
```

### 5. Templates and Jinja2
```html
<!-- base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        <a href="{{ url_for('about') }}">About</a>
    </nav>
    
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            {% for message in messages %}
                <div class="alert">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    {% block content %}{% endblock %}
</body>
</html>

<!-- index.html -->
{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
<h1>Welcome to Flask</h1>
<p>Current time: {{ current_time }}</p>

{% if user %}
    <p>Hello, {{ user.name }}!</p>
{% else %}
    <p>Please <a href="{{ url_for('login') }}">login</a></p>
{% endif %}

<ul>
{% for item in items %}
    <li>{{ item }}</li>
{% endfor %}
</ul>
{% endblock %}
```

### 6. Forms and Validation
```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Process login
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)
```

### 7. Sessions and Cookies
```python
from flask import session, request

@app.route('/set_session')
def set_session():
    session['user_id'] = 123
    session['username'] = 'john_doe'
    return 'Session data set'

@app.route('/get_session')
def get_session():
    user_id = session.get('user_id')
    username = session.get('username')
    return f'User ID: {user_id}, Username: {username}'

@app.route('/clear_session')
def clear_session():
    session.clear()
    return 'Session cleared'
```

## Advanced Topics

### 1. Application Factory Pattern
```python
def create_app(config_name='default'):
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Register blueprints
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
```

### 2. Blueprints for Modular Applications
```python
# main/__init__.py
from flask import Blueprint

main = Blueprint('main', __name__)

from . import views

# main/views.py
from flask import render_template
from . import main

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')
```

### 3. Database Integration with SQLAlchemy
```python
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    def __repr__(self):
        return f'<User {self.username}>'

@app.route('/users')
def users():
    users = User.query.all()
    return render_template('users.html', users=users)
```

### 4. RESTful API Development
```python
from flask_restful import Api, Resource

api = Api(app)

class UserAPI(Resource):
    def get(self, user_id):
        user = User.query.get_or_404(user_id)
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
    
    def put(self, user_id):
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        user.username = data.get('username', user.username)
        user.email = data.get('email', user.email)
        db.session.commit()
        return {'message': 'User updated successfully'}
    
    def delete(self, user_id):
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        return {'message': 'User deleted successfully'}

api.add_resource(UserAPI, '/api/users/<int:user_id>')
```

### 5. Error Handling
```python
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.errorhandler(ValidationError)
def validation_error(error):
    return jsonify({'error': 'Validation failed', 'details': str(error)}), 400
```

### 6. Authentication and Authorization
```python
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    # User model implementation
    pass

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')
```

## Best Practices

### 1. Configuration Management
```python
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### 2. Security Best Practices
```python
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

csrf = CSRFProtect()
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/data')
@limiter.limit("10 per minute")
def api_data():
    return jsonify({'data': 'sensitive data'})
```

### 3. Testing
```python
import unittest
from app import create_app, db

class TestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
```

### 4. Deployment
```python
# requirements.txt
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Migrate==4.0.5
Flask-Login==0.6.3
Flask-WTF==1.1.1
gunicorn==21.2.0

# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## Common Patterns

### 1. API Response Formatting
```python
def api_response(data=None, message="Success", status_code=200):
    response = {
        'status': 'success' if status_code < 400 else 'error',
        'message': message,
        'data': data
    }
    return jsonify(response), status_code

@app.route('/api/users')
def get_users():
    users = User.query.all()
    user_data = [{'id': u.id, 'username': u.username} for u in users]
    return api_response(data=user_data)
```

### 2. Pagination
```python
def paginate(query, page, per_page=20):
    pagination = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    return {
        'items': [item.to_dict() for item in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev
    }
```

### 3. Background Tasks
```python
from celery import Celery

celery = Celery('app', broker='redis://localhost:6379')

@celery.task
def send_email(recipient, subject, body):
    # Send email logic
    pass

@app.route('/send-email')
def send_email_route():
    send_email.delay('user@example.com', 'Subject', 'Body')
    return 'Email queued'
```

## Quick Checks

### Check 1: Basic Route
```python
# What will this return?
@app.route('/hello/<name>')
def hello(name):
    return f'Hello, {name}!'

# URL: /hello/World
```

### Check 2: Template Variables
```python
# In template: {{ user.name if user else 'Guest' }}
# What will be displayed if user is None?
```

### Check 3: Form Validation
```python
# What happens if form validation fails?
if form.validate_on_submit():
    # Process form
    pass
# Where does execution continue?
```

## Lab Problems

### Lab 1: Blog Application
Create a complete blog application with user authentication, post creation, and commenting.

### Lab 2: RESTful API
Build a RESTful API for a task management system with CRUD operations.

### Lab 3: E-commerce Site
Develop an e-commerce site with product catalog, shopping cart, and order management.

### Lab 4: Real-time Chat
Implement a real-time chat application using WebSockets and Flask-SocketIO.

## AI Code Comparison
When working with AI-generated Flask code, evaluate:
- **Security considerations** - are CSRF tokens and input validation implemented?
- **Error handling** - are all potential errors caught and handled appropriately?
- **Database relationships** - are foreign keys and relationships properly defined?
- **API design** - are RESTful principles followed consistently?
- **Performance** - are database queries optimized and caching implemented?

## Next Steps
- Learn about Flask extensions and ecosystem
- Master database design and ORM patterns
- Explore microservices architecture
- Study containerization and deployment strategies
- Understand web security and authentication systems
