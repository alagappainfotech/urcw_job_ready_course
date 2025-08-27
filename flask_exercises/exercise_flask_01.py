####################################################
## Flask Exercise 1: Basic Flask Application
####################################################

# Complete the following tasks to create a basic Flask application

"""
Exercise 1: Create a Simple Flask Application

Tasks:
1. Create a Flask app with routes for home, about, and contact
2. Create templates for each page
3. Add navigation between pages
4. Style the pages with CSS

Steps:
1. Install Flask: pip install flask
2. Create app.py
3. Create templates directory
4. Create static directory for CSS
5. Run the application
"""

# Create app.py:
"""
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

@app.route('/')
def home():
    return render_template('index.html', current_time=datetime.now())

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
"""

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
        {% block content %}{% endblock %}
    </main>
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
<p>This is your first Flask application!</p>
{% endblock %}
"""

# Create templates/about.html:
"""
{% extends "base.html" %}

{% block title %}About{% endblock %}

{% block content %}
<h1>About Us</h1>
<p>This is a simple Flask application created for learning purposes.</p>
<p>Flask is a lightweight web framework for Python.</p>
{% endblock %}
"""

# Create templates/contact.html:
"""
{% extends "base.html" %}

{% block title %}Contact{% endblock %}

{% block content %}
<h1>Contact Us</h1>
<p>Email: contact@example.com</p>
<p>Phone: (123) 456-7890</p>
<p>Address: 123 Main St, City, State 12345</p>
{% endblock %}
"""

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

nav a:hover {
    color: #ddd;
}

main {
    max-width: 800px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
}
"""

"""
Exercise 2: Create a Todo List Application

Tasks:
1. Create routes for listing, adding, and completing todos
2. Use sessions to store todos
3. Create forms for adding todos
4. Add ability to mark todos as complete

Steps:
1. Add todo routes to app.py
2. Create todo templates
3. Add session management
4. Create todo form
"""

# Update app.py to include todo functionality:
"""
from flask import Flask, render_template, request, redirect, url_for, flash, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize todos in session
def get_todos():
    if 'todos' not in session:
        session['todos'] = []
    return session['todos']

@app.route('/')
def home():
    return render_template('index.html', current_time=datetime.now())

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/todos')
def todo_list():
    todos = get_todos()
    return render_template('todos.html', todos=todos)

@app.route('/todos/add', methods=['GET', 'POST'])
def add_todo():
    if request.method == 'POST':
        title = request.form.get('title')
        if title:
            todos = get_todos()
            todo = {
                'id': len(todos) + 1,
                'title': title,
                'completed': False,
                'created_at': datetime.now().isoformat()
            }
            todos.append(todo)
            session['todos'] = todos
            flash('Todo added successfully!', 'success')
            return redirect(url_for('todo_list'))
        else:
            flash('Title is required!', 'error')
    
    return render_template('add_todo.html')

@app.route('/todos/<int:todo_id>/toggle')
def toggle_todo(todo_id):
    todos = get_todos()
    for todo in todos:
        if todo['id'] == todo_id:
            todo['completed'] = not todo['completed']
            session['todos'] = todos
            flash('Todo updated!', 'success')
            break
    
    return redirect(url_for('todo_list'))

@app.route('/todos/<int:todo_id>/delete')
def delete_todo(todo_id):
    todos = get_todos()
    todos = [todo for todo in todos if todo['id'] != todo_id]
    session['todos'] = todos
    flash('Todo deleted!', 'success')
    return redirect(url_for('todo_list'))
"""

# Create templates/todos.html:
"""
{% extends "base.html" %}

{% block title %}Todo List{% endblock %}

{% block content %}
<h1>Todo List</h1>

{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="alert alert-{{ category }}">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}

<a href="{{ url_for('add_todo') }}" class="btn btn-primary">Add New Todo</a>

{% if todos %}
    <ul class="todo-list">
        {% for todo in todos %}
            <li class="todo-item {% if todo.completed %}completed{% endif %}">
                <span class="todo-title">{{ todo.title }}</span>
                <div class="todo-actions">
                    <a href="{{ url_for('toggle_todo', todo_id=todo.id) }}" class="btn btn-sm btn-{% if todo.completed %}warning{% else %}success{% endif %}">
                        {% if todo.completed %}Mark Incomplete{% else %}Mark Complete{% endif %}
                    </a>
                    <a href="{{ url_for('delete_todo', todo_id=todo.id) }}" class="btn btn-sm btn-danger" onclick="return confirm('Are you sure?')">Delete</a>
                </div>
            </li>
        {% endfor %}
    </ul>
{% else %}
    <p>No todos yet. <a href="{{ url_for('add_todo') }}">Add your first todo!</a></p>
{% endif %}
{% endblock %}
"""

# Create templates/add_todo.html:
"""
{% extends "base.html" %}

{% block title %}Add Todo{% endblock %}

{% block content %}
<h1>Add New Todo</h1>

<form method="post">
    <div class="form-group">
        <label for="title">Todo Title:</label>
        <input type="text" id="title" name="title" class="form-control" required>
    </div>
    <button type="submit" class="btn btn-primary">Add Todo</button>
    <a href="{{ url_for('todo_list') }}" class="btn btn-secondary">Cancel</a>
</form>
{% endblock %}
"""

# Update static/css/style.css to include todo styles:
"""
/* Add these styles to your existing CSS */

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

.btn {
    display: inline-block;
    padding: 8px 16px;
    text-decoration: none;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    margin: 2px;
}

.btn-primary {
    background-color: #007bff;
    color: white;
}

.btn-secondary {
    background-color: #6c757d;
    color: white;
}

.btn-success {
    background-color: #28a745;
    color: white;
}

.btn-warning {
    background-color: #ffc107;
    color: #212529;
}

.btn-danger {
    background-color: #dc3545;
    color: white;
}

.btn-sm {
    padding: 4px 8px;
    font-size: 12px;
}

.todo-list {
    list-style: none;
    padding: 0;
}

.todo-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    margin: 5px 0;
    background-color: #f8f9fa;
    border-radius: 4px;
    border-left: 4px solid #007bff;
}

.todo-item.completed {
    opacity: 0.6;
    border-left-color: #28a745;
}

.todo-item.completed .todo-title {
    text-decoration: line-through;
}

.todo-title {
    flex-grow: 1;
    margin-right: 10px;
}

.todo-actions {
    display: flex;
    gap: 5px;
}

.form-group {
    margin-bottom: 15px;
}

.form-control {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
}
"""

"""
Exercise 3: Create a Simple API

Tasks:
1. Create API endpoints for CRUD operations
2. Use JSON for request/response
3. Add proper error handling
4. Implement data validation

Steps:
1. Add API routes to app.py
2. Create JSON response functions
3. Add error handling
4. Test with curl or Postman
"""
"""
# Add API functionality to app.py:
"""
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime

# ... existing code ...

# API Routes
@app.route('/api/todos', methods=['GET'])
def api_get_todos():
    """Get all todos"""
    todos = get_todos()
    return jsonify({
        'success': True,
        'data': todos,
        'count': len(todos)
    })

@app.route('/api/todos', methods=['POST'])
def api_create_todo():
    """Create a new todo"""
    data = request.get_json()
    
    if not data or 'title' not in data:
        return jsonify({
            'success': False,
            'error': 'Title is required'
        }), 400
    
    todos = get_todos()
    todo = {
        'id': len(todos) + 1,
        'title': data['title'],
        'completed': False,
        'created_at': datetime.now().isoformat()
    }
    todos.append(todo)
    session['todos'] = todos
    
    return jsonify({
        'success': True,
        'data': todo
    }), 201

@app.route('/api/todos/<int:todo_id>', methods=['GET'])
def api_get_todo(todo_id):
    """Get a specific todo"""
    todos = get_todos()
    todo = next((t for t in todos if t['id'] == todo_id), None)
    
    if not todo:
        return jsonify({
            'success': False,
            'error': 'Todo not found'
        }), 404
    
    return jsonify({
        'success': True,
        'data': todo
    })

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
def api_update_todo(todo_id):
    """Update a todo"""
    todos = get_todos()
    todo = next((t for t in todos if t['id'] == todo_id), None)
    
    if not todo:
        return jsonify({
            'success': False,
            'error': 'Todo not found'
        }), 404
    
    data = request.get_json()
    
    if 'title' in data:
        todo['title'] = data['title']
    if 'completed' in data:
        todo['completed'] = data['completed']
    
    session['todos'] = todos
    
    return jsonify({
        'success': True,
        'data': todo
    })

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def api_delete_todo(todo_id):
    """Delete a todo"""
    todos = get_todos()
    todo = next((t for t in todos if t['id'] == todo_id), None)
    
    if not todo:
        return jsonify({
            'success': False,
            'error': 'Todo not found'
        }), 404
    
    todos = [t for t in todos if t['id'] != todo_id]
    session['todos'] = todos
    
    return jsonify({
        'success': True,
        'message': 'Todo deleted successfully'
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


"""
Exercise 4: Add User Authentication

Tasks:
1. Create login and registration forms
2. Implement session-based authentication
3. Add protected routes
4. Create user profile pages

Steps:
1. Add user management to app.py
2. Create authentication templates
3. Add login/logout functionality
4. Protect routes with decorators
"""

# Add authentication to app.py:
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Simple user storage (in production, use a database)
users = {
    'admin': {'password': 'password123', 'email': 'admin@example.com'}
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username]['password'] == password:
            session['user_id'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if username in users:
            flash('Username already exists.', 'error')
        else:
            users[username] = {'password': password, 'email': email}
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/profile')
@login_required
def profile():
    username = session['user_id']
    user = users[username]
    return render_template('profile.html', username=username, user=user)

# Update existing routes to require login
@app.route('/todos')
@login_required
def todo_list():
    todos = get_todos()
    return render_template('todos.html', todos=todos)

@app.route('/todos/add', methods=['GET', 'POST'])
@login_required
def add_todo():
    # ... existing code ...


# Create templates/login.html:
"""
{% extends "base.html" %}

{% block title %}Login{% endblock %}

{% block content %}
<h1>Login</h1>

{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="alert alert-{{ category }}">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}

<form method="post">
    <div class="form-group">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" class="form-control" required>
    </div>
    <div class="form-group">
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" class="form-control" required>
    </div>
    <button type="submit" class="btn btn-primary">Login</button>
</form>

<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
{% endblock %}
"""

# Create templates/register.html:
"""
{% extends "base.html" %}

{% block title %}Register{% endblock %}

{% block content %}
<h1>Register</h1>

{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="alert alert-{{ category }}">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}

<form method="post">
    <div class="form-group">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" class="form-control" required>
    </div>
    <div class="form-group">
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" class="form-control" required>
    </div>
    <div class="form-group">
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" class="form-control" required>
    </div>
    <button type="submit" class="btn btn-primary">Register</button>
</form>

<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
{% endblock %}
"""

# Create templates/profile.html:
"""
{% extends "base.html" %}

{% block title %}Profile{% endblock %}

{% block content %}
<h1>Profile</h1>

<p><strong>Username:</strong> {{ username }}</p>
<p><strong>Email:</strong> {{ user.email }}</p>

<a href="{{ url_for('logout') }}" class="btn btn-danger">Logout</a>
{% endblock %}
"""

# Update templates/base.html to include login/logout links:
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
        {% if session.user_id %}
            <a href="{{ url_for('todo_list') }}">Todos</a>
            <a href="{{ url_for('profile') }}">Profile</a>
            <a href="{{ url_for('logout') }}">Logout</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a>
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
"""

# Test your Flask application:
"""
1. Run the application: python app.py
2. Visit http://127.0.0.1:5000/ to see your home page
3. Test the todo list at /todos/
4. Test user registration at /register/
5. Test login at /login/
6. Test the API endpoints:
   - GET /api/todos
   - POST /api/todos
   - GET /api/todos/1
   - PUT /api/todos/1
   - DELETE /api/todos/1
""" 