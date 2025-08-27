####################################################
## Flask Framework - Lesson 1: Introduction
####################################################

# Flask is a lightweight web framework for Python
# It's known for its simplicity and flexibility
# Flask follows the WSGI (Web Server Gateway Interface) standard

# 1. Flask Philosophy and Features

"""
Flask Features:
- Lightweight and minimal
- Flexible and extensible
- Jinja2 templating engine
- Built-in development server
- RESTful request handling
- Easy to learn and use
- Large ecosystem of extensions
"""

# 2. Setting Up a Flask Project

# First, install Flask:
# pip install flask

# Basic Flask application structure:
"""
myapp/
├── app.py              # Main application file
├── templates/          # HTML templates
│   ├── base.html
│   └── index.html
├── static/            # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── requirements.txt    # Python dependencies
└── config.py          # Configuration file
"""

# 3. Your First Flask Application

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from datetime import datetime

# Create Flask application instance
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for sessions and flash messages

# 4. Basic Routes

@app.route('/')
def home():
    """Home page route"""
    return "Hello, Flask World!"

@app.route('/about')
def about():
    """About page route"""
    return "This is the About page"

@app.route('/user/<username>')
def show_user(username):
    """Route with parameter"""
    return f"Hello, {username}!"

@app.route('/post/<int:post_id>')
def show_post(post_id):
    """Route with integer parameter"""
    return f"Post ID: {post_id}"

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    """Route with path parameter"""
    return f"Subpath: {subpath}"

# 5. HTTP Methods

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle both GET and POST requests"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'password':
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    """API endpoint example"""
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({'message': 'Data received', 'data': data})
    else:
        return jsonify({'message': 'GET request', 'timestamp': datetime.now().isoformat()})

# 6. Templates and Jinja2

# Create templates/base.html:
"""
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('about') }}">About</a>
        <a href="{{ url_for('contact') }}">Contact</a>
    </nav>
    
    <main>
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </main>
    
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
"""

# Create templates/index.html:
"""
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
"""

# 7. Rendering Templates

@app.route('/')
def home():
    """Home page with template"""
    context = {
        'current_time': datetime.now(),
        'user': {'name': 'John Doe'},
        'items': ['Item 1', 'Item 2', 'Item 3']
    }
    return render_template('index.html', **context)

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

# 8. Forms and Request Handling

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email

class LoginForm(FlaskForm):
    """Login form using Flask-WTF"""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login with form validation"""
    form = LoginForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        if username == 'admin' and password == 'password':
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html', form=form)

# 9. Static Files

# Create static/css/style.css:
"""
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f4f4f4;
}

nav {
    background-color: #333;
    padding: 10px;
    margin-bottom: 20px;
}

nav a {
    color: white;
    text-decoration: none;
    margin-right: 20px;
}

.alert {
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.alert-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.alert-error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
"""

# 10. Configuration

# Create config.py:
"""
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
"""

# 11. Application Factory Pattern

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    from config import config
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    # db.init_app(app)
    # login_manager.init_app(app)
    
    # Register blueprints
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app

# 12. Blueprints

# Create main/__init__.py:
"""
from flask import Blueprint

main = Blueprint('main', __name__)

from . import views
"""

# Create main/views.py:
"""
from flask import render_template, request, jsonify
from . import main

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/api/status')
def api_status():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
"""

# 13. Error Handling

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# 14. Sessions and Cookies

from flask import session

@app.route('/set_session')
def set_session():
    """Set session data"""
    session['user_id'] = 123
    session['username'] = 'john_doe'
    return "Session data set"

@app.route('/get_session')
def get_session():
    """Get session data"""
    user_id = session.get('user_id')
    username = session.get('username')
    return f"User ID: {user_id}, Username: {username}"

@app.route('/clear_session')
def clear_session():
    """Clear session data"""
    session.clear()
    return "Session cleared"

# 15. Practical Example: Simple Blog

class Post:
    """Simple post class (in real app, use database)"""
    def __init__(self, id, title, content, author):
        self.id = id
        self.title = title
        self.content = content
        self.author = author
        self.created_at = datetime.now()

# Sample data
posts = [
    Post(1, "First Post", "This is the first post content.", "John"),
    Post(2, "Second Post", "This is the second post content.", "Jane"),
]

@app.route('/blog')
def blog():
    """Blog listing page"""
    return render_template('blog.html', posts=posts)

@app.route('/blog/<int:post_id>')
def blog_post(post_id):
    """Individual blog post"""
    post = next((p for p in posts if p.id == post_id), None)
    if post is None:
        abort(404)
    return render_template('blog_post.html', post=post)

@app.route('/blog/new', methods=['GET', 'POST'])
def new_post():
    """Create new blog post"""
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        author = request.form['author']
        
        new_post = Post(len(posts) + 1, title, content, author)
        posts.append(new_post)
        
        flash('Post created successfully!', 'success')
        return redirect(url_for('blog'))
    
    return render_template('new_post.html')

# 16. API Development

@app.route('/api/posts', methods=['GET'])
def api_posts():
    """API endpoint for posts"""
    return jsonify([{
        'id': post.id,
        'title': post.title,
        'author': post.author,
        'created_at': post.created_at.isoformat()
    } for post in posts])

@app.route('/api/posts/<int:post_id>', methods=['GET'])
def api_post(post_id):
    """API endpoint for single post"""
    post = next((p for p in posts if p.id == post_id), None)
    if post is None:
        return jsonify({'error': 'Post not found'}), 404
    
    return jsonify({
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author,
        'created_at': post.created_at.isoformat()
    })

@app.route('/api/posts', methods=['POST'])
def api_create_post():
    """API endpoint to create post"""
    data = request.get_json()
    
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    new_post = Post(len(posts) + 1, data['title'], data['content'], data.get('author', 'Anonymous'))
    posts.append(new_post)
    
    return jsonify({
        'id': new_post.id,
        'title': new_post.title,
        'author': new_post.author,
        'created_at': new_post.created_at.isoformat()
    }), 201

# 17. Exercises

# Exercise 1: Create a simple Flask application
"""
1. Create a Flask app with routes for home, about, and contact
2. Create templates for each page
3. Add navigation between pages
4. Style the pages with CSS
"""

# Exercise 2: Create a todo list application
"""
1. Create routes for listing, adding, and completing todos
2. Use sessions to store todos
3. Create forms for adding todos
4. Add ability to mark todos as complete
"""

# Exercise 3: Create a simple API
"""
1. Create API endpoints for CRUD operations
2. Use JSON for request/response
3. Add proper error handling
4. Implement data validation
"""

# Exercise 4: Add user authentication
"""
1. Create login and registration forms
2. Implement session-based authentication
3. Add protected routes
4. Create user profile pages
"""

# 18. Best Practices

"""
1. Use application factory pattern for larger apps
2. Organize code with blueprints
3. Use environment variables for configuration
4. Implement proper error handling
5. Validate all user input
6. Use CSRF protection for forms
7. Keep templates simple and reusable
8. Use static files for CSS, JS, and images
9. Write tests for your application
10. Use proper HTTP status codes
"""

# 19. Common Flask Extensions

"""
Flask-SQLAlchemy: Database ORM
Flask-Login: User session management
Flask-WTF: Form handling and validation
Flask-Mail: Email support
Flask-Migrate: Database migrations
Flask-RESTful: REST API development
Flask-CORS: Cross-origin resource sharing
Flask-JWT-Extended: JWT authentication
"""

# 20. Running the Application

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Next Steps:
# - Learn about Flask-SQLAlchemy for database integration
# - Understand Flask-Login for authentication
# - Master Flask-WTF for form handling
# - Learn about Flask blueprints and application structure
# - Explore Flask extensions and ecosystem 