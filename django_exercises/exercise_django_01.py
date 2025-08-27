####################################################
## Django Exercise 1: Basic Django Project
####################################################

# Complete the following tasks to create a basic Django project

"""
Exercise 1: Create a Simple Django Project

Tasks:
1. Create a new Django project called 'myblog'
2. Create a new app called 'blog'
3. Add the blog app to INSTALLED_APPS
4. Create a simple view that returns "Hello, Django!"
5. Configure URL routing
6. Run the development server

Steps:
1. django-admin startproject myblog
2. cd myblog
3. python manage.py startapp blog
4. Add 'blog' to INSTALLED_APPS in settings.py
5. Create views.py in blog app
6. Create urls.py in blog app
7. Update project urls.py
8. python manage.py runserver
"""

# Your Django project structure should look like this:
"""
myblog/
├── manage.py
├── myblog/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── blog/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── models.py
    ├── tests.py
    ├── urls.py
    └── views.py
"""

# In blog/views.py, add:
"""
from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return HttpResponse("Hello, Django!")

def about(request):
    return HttpResponse("This is the About page")

def contact(request):
    return HttpResponse("Contact us at contact@example.com")
"""

# In blog/urls.py, add:
"""
from django.urls import path
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]
"""

# In myblog/urls.py, add:
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('blog.urls')),
]
"""

"""
Exercise 2: Create a Contact Form

Tasks:
1. Create a Contact model with name, email, and message fields
2. Create a form for the Contact model
3. Create a view to handle form submission
4. Create a template to display the form
5. Add success message after form submission

Steps:
1. Add Contact model to blog/models.py
2. Create ContactForm in blog/forms.py
3. Create contact view in blog/views.py
4. Create contact.html template
5. Add URL pattern
6. Run migrations
"""

# In blog/models.py, add:
"""
from django.db import models

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.email}"
"""

# In blog/forms.py, add:
"""
from django import forms
from .models import Contact

class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'message']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
        }
"""

# In blog/views.py, add:
"""
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import ContactForm

def contact_form(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your message has been sent successfully!')
            return redirect('blog:contact_form')
    else:
        form = ContactForm()
    
    return render(request, 'blog/contact_form.html', {'form': form})
"""

# Create blog/templates/blog/contact_form.html:
"""
{% extends 'base.html' %}

{% block content %}
<h1>Contact Us</h1>

{% if messages %}
    {% for message in messages %}
        <div class="alert alert-{{ message.tags }}">
            {{ message }}
        </div>
    {% endfor %}
{% endif %}

<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Send Message</button>
</form>
{% endblock %}
"""

"""
Exercise 3: Create a Todo List Application

Tasks:
1. Create a Todo model with title, description, and completed fields
2. Create views to list, create, and mark todos as complete
3. Create templates for listing and creating todos
4. Add functionality to mark todos as complete/incomplete

Steps:
1. Add Todo model to blog/models.py
2. Create TodoForm in blog/forms.py
3. Create todo views in blog/views.py
4. Create todo templates
5. Add URL patterns
6. Run migrations
"""

# In blog/models.py, add:
"""
class Todo(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created_at']
"""

# In blog/forms.py, add:
"""
from .models import Todo

class TodoForm(forms.ModelForm):
    class Meta:
        model = Todo
        fields = ['title', 'description']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }
"""

# In blog/views.py, add:
"""
from .models import Todo
from .forms import TodoForm

def todo_list(request):
    todos = Todo.objects.all()
    return render(request, 'blog/todo_list.html', {'todos': todos})

def todo_create(request):
    if request.method == 'POST':
        form = TodoForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Todo created successfully!')
            return redirect('blog:todo_list')
    else:
        form = TodoForm()
    
    return render(request, 'blog/todo_form.html', {'form': form})

def todo_toggle(request, pk):
    todo = get_object_or_404(Todo, pk=pk)
    todo.completed = not todo.completed
    todo.save()
    return redirect('blog:todo_list')
"""

# Create blog/templates/blog/todo_list.html:
"""
{% extends 'base.html' %}

{% block content %}
<h1>Todo List</h1>

<a href="{% url 'blog:todo_create' %}" class="btn btn-primary mb-3">Add New Todo</a>

{% for todo in todos %}
    <div class="card mb-2">
        <div class="card-body">
            <h5 class="card-title {% if todo.completed %}text-muted text-decoration-line-through{% endif %}">
                {{ todo.title }}
            </h5>
            <p class="card-text">{{ todo.description }}</p>
            <a href="{% url 'blog:todo_toggle' todo.pk %}" class="btn btn-sm btn-{% if todo.completed %}warning{% else %}success{% endif %}">
                {% if todo.completed %}Mark Incomplete{% else %}Mark Complete{% endif %}
            </a>
        </div>
    </div>
{% empty %}
    <p>No todos yet.</p>
{% endfor %}
{% endblock %}
"""

# Create blog/templates/blog/todo_form.html:
"""
{% extends 'base.html' %}

{% block content %}
<h1>Create New Todo</h1>

<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Create Todo</button>
    <a href="{% url 'blog:todo_list' %}" class="btn btn-secondary">Cancel</a>
</form>
{% endblock %}
"""

"""
Exercise 4: Add User Authentication

Tasks:
1. Create login and registration forms
2. Implement session-based authentication
3. Add protected routes
4. Create user profile pages

Steps:
1. Create UserRegistrationForm in blog/forms.py
2. Create authentication views in blog/views.py
3. Create login and registration templates
4. Add URL patterns
5. Add login_required decorators
"""

# In blog/forms.py, add:
"""
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
"""

# In blog/views.py, add:
"""
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('blog:home')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'blog/register.html', {'form': form})

@login_required
def profile(request):
    return render(request, 'blog/profile.html', {'user': request.user})
"""

# Create blog/templates/blog/register.html:
"""
{% extends 'base.html' %}

{% block content %}
<h1>Register</h1>

<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Register</button>
</form>
{% endblock %}
"""

# Create blog/templates/blog/profile.html:
"""
{% extends 'base.html' %}

{% block content %}
<h1>Profile</h1>

<p>Username: {{ user.username }}</p>
<p>Email: {{ user.email }}</p>
<p>Date joined: {{ user.date_joined }}</p>
{% endblock %}
"""

# Update blog/urls.py:
"""
from django.urls import path
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('contact/form/', views.contact_form, name='contact_form'),
    path('todos/', views.todo_list, name='todo_list'),
    path('todos/create/', views.todo_create, name='todo_create'),
    path('todos/<int:pk>/toggle/', views.todo_toggle, name='todo_toggle'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
]
"""

# Don't forget to run migrations:
"""
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
"""

# Test your Django application:
"""
1. Run the development server: python manage.py runserver
2. Visit http://127.0.0.1:8000/ to see your home page
3. Test the contact form at /contact/form/
4. Test the todo list at /todos/
5. Test user registration at /register/
6. Test the admin interface at /admin/
""" 