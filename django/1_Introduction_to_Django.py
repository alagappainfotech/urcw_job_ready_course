####################################################
## Django Framework - Lesson 1: Introduction
####################################################

# Django is a high-level Python web framework that encourages rapid development
# and clean, pragmatic design. It follows the MVT (Model-View-Template) pattern.

# 1. Django Philosophy and Features

"""
Django Features:
- ORM (Object-Relational Mapping)
- Admin interface
- URL routing
- Template system
- Form handling
- Security features
- Database migrations
- Built-in user authentication
"""

# 2. Setting Up a Django Project

# First, install Django:
# pip install django

# Create a new Django project:
# django-admin startproject myproject
# cd myproject

# Create a new Django app:
# python manage.py startapp myapp

# 3. Project Structure (after running the above commands)

"""
myproject/
├── manage.py              # Django's command-line utility
├── myproject/             # Project configuration directory
│   ├── __init__.py
│   ├── settings.py        # Project settings
│   ├── urls.py           # URL declarations
│   ├── wsgi.py           # WSGI application entry point
│   └── asgi.py           # ASGI application entry point
└── myapp/                # Application directory
    ├── __init__.py
    ├── admin.py          # Admin interface configuration
    ├── apps.py           # Application configuration
    ├── models.py         # Database models
    ├── tests.py          # Unit tests
    ├── urls.py           # Application URL routing
    └── views.py          # View functions/classes
"""

# 4. Basic Django Settings

# settings.py example (key configurations):
"""
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # Add your app here
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
"""

# 5. Creating Your First View

# In myapp/views.py:
"""
from django.shortcuts import render
from django.http import HttpResponse

def hello_world(request):
    return HttpResponse("Hello, Django World!")

def home(request):
    context = {
        'title': 'Welcome to Django',
        'message': 'This is your first Django app!'
    }
    return render(request, 'myapp/home.html', context)
"""

# 6. URL Configuration

# In myproject/urls.py (project-level URLs):
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),  # Include app URLs
]
"""

# In myapp/urls.py (app-level URLs):
"""
from django.urls import path
from . import views

app_name = 'myapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('hello/', views.hello_world, name='hello'),
]
"""

# 7. Creating Templates

# Create directory: myapp/templates/myapp/
# Create file: myapp/templates/myapp/home.html

"""
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>{{ message }}</p>
    <a href="{% url 'myapp:hello' %}">Say Hello</a>
</body>
</html>
"""

# 8. Running the Development Server

# Commands to run:
"""
python manage.py migrate          # Apply database migrations
python manage.py runserver        # Start development server
python manage.py createsuperuser  # Create admin user
"""

# 9. Django Admin Interface

# In myapp/admin.py:
"""
from django.contrib import admin
from .models import YourModel

admin.site.register(YourModel)
"""

# 10. Basic Models

# In myapp/models.py:
"""
from django.db import models
from django.contrib.auth.models import User

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created_date']
"""

# 11. Forms in Django

# In myapp/forms.py:
"""
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
        }
"""

# 12. Practical Example: Simple Blog

# Complete example combining all concepts:

# myapp/models.py
"""
from django.db import models
from django.contrib.auth.models import User

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
"""

# myapp/views.py
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import BlogPost
from .forms import BlogPostForm

def blog_list(request):
    posts = BlogPost.objects.all()
    return render(request, 'myapp/blog_list.html', {'posts': posts})

def blog_detail(request, pk):
    post = get_object_or_404(BlogPost, pk=pk)
    return render(request, 'myapp/blog_detail.html', {'post': post})

@login_required
def blog_create(request):
    if request.method == 'POST':
        form = BlogPostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            return redirect('myapp:blog_list')
    else:
        form = BlogPostForm()
    return render(request, 'myapp/blog_form.html', {'form': form})
"""

# myapp/forms.py
"""
from django import forms
from .models import BlogPost

class BlogPostForm(forms.ModelForm):
    class Meta:
        model = BlogPost
        fields = ['title', 'content']
"""

# myapp/urls.py
"""
from django.urls import path
from . import views

app_name = 'myapp'

urlpatterns = [
    path('', views.blog_list, name='blog_list'),
    path('post/<int:pk>/', views.blog_detail, name='blog_detail'),
    path('post/new/', views.blog_create, name='blog_create'),
]
"""

# 13. Template Examples

# myapp/templates/myapp/blog_list.html
"""
{% extends 'base.html' %}

{% block content %}
<h1>Blog Posts</h1>
{% if user.is_authenticated %}
    <a href="{% url 'myapp:blog_create' %}" class="btn btn-primary">New Post</a>
{% endif %}

{% for post in posts %}
    <article>
        <h2><a href="{% url 'myapp:blog_detail' post.pk %}">{{ post.title }}</a></h2>
        <p>By {{ post.author }} on {{ post.created_date }}</p>
        <p>{{ post.content|truncatewords:30 }}</p>
    </article>
{% empty %}
    <p>No posts yet.</p>
{% endfor %}
{% endblock %}
"""

# 14. Exercises

# Exercise 1: Create a simple Django project with a "Hello World" view
"""
1. Create a new Django project
2. Create a new app
3. Add a view that returns "Hello, Django!"
4. Configure URLs to display this view
5. Run the development server
"""

# Exercise 2: Create a simple contact form
"""
1. Create a Contact model with name, email, and message fields
2. Create a form for the Contact model
3. Create a view to handle form submission
4. Create a template to display the form
5. Add success message after form submission
"""

# Exercise 3: Create a simple todo list
"""
1. Create a Todo model with title, description, and completed fields
2. Create views to list, create, and mark todos as complete
3. Create templates for listing and creating todos
4. Add functionality to mark todos as complete/incomplete
"""

# Exercise 4: Add user authentication
"""
1. Create login and logout views
2. Create registration form
3. Add user profile page
4. Protect views that require authentication
"""

# 15. Best Practices

"""
1. Always use virtual environments
2. Keep settings organized (use environment variables)
3. Use meaningful model names and field names
4. Write tests for your views and models
5. Use Django's built-in security features
6. Follow PEP 8 coding standards
7. Use Django's ORM instead of raw SQL when possible
8. Keep templates simple and reusable
9. Use Django's form validation
10. Document your code
"""

# 16. Common Django Commands

"""
django-admin startproject projectname    # Create new project
python manage.py startapp appname       # Create new app
python manage.py makemigrations         # Create database migrations
python manage.py migrate                # Apply migrations
python manage.py runserver              # Start development server
python manage.py createsuperuser       # Create admin user
python manage.py collectstatic          # Collect static files
python manage.py test                   # Run tests
python manage.py shell                  # Open Django shell
"""

# Next Steps:
# - Learn about Django Models and Database relationships
# - Understand Django Forms and validation
# - Master Django Templates and template inheritance
# - Learn about Django Class-Based Views
# - Explore Django REST Framework for APIs 