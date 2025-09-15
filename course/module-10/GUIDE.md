# Module 10: Web Development with Django - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Build scalable web applications using Django framework
- Implement the MVT (Model-View-Template) architecture pattern
- Create and manage Django models with relationships
- Build RESTful APIs using Django REST Framework
- Implement user authentication and authorization
- Deploy Django applications to production
- Apply Django best practices and security measures

## Core Concepts

### 1. Django Framework Overview
Django is a high-level Python web framework that provides:
- **MVT Architecture** (Model-View-Template)
- **ORM** (Object-Relational Mapping) for database operations
- **Admin Interface** for content management
- **URL Routing** with regular expressions
- **Template System** with inheritance and filters
- **Form Handling** with validation
- **Security Features** built-in
- **Scalability** for large applications

### 2. Django Project Structure
```
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
```

### 3. Models and Database
```python
from django.db import models
from django.contrib.auth.models import User

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    published = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_date']
        verbose_name = 'Blog Post'
        verbose_name_plural = 'Blog Posts'
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('post_detail', kwargs={'pk': self.pk})

class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_date = models.DateTimeField(auto_now_add=True)
    approved = models.BooleanField(default=False)
    
    def __str__(self):
        return f'Comment by {self.author} on {self.post.title}'
```

### 4. Views and URL Patterns
```python
# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from .models import Post, Comment
from .forms import PostForm, CommentForm

# Function-based views
def post_list(request):
    posts = Post.objects.filter(published=True)
    return render(request, 'blog/post_list.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    comments = post.comments.filter(approved=True)
    
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.post = post
            comment.author = request.user
            comment.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = CommentForm()
    
    return render(request, 'blog/post_detail.html', {
        'post': post,
        'comments': comments,
        'form': form
    })

# Class-based views
class PostListView(ListView):
    model = Post
    template_name = 'blog/post_list.html'
    context_object_name = 'posts'
    paginate_by = 10
    
    def get_queryset(self):
        return Post.objects.filter(published=True)

class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/post_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comments'] = self.object.comments.filter(approved=True)
        context['form'] = CommentForm()
        return context

class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    form_class = PostForm
    template_name = 'blog/post_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

# urls.py
from django.urls import path, include
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.PostListView.as_view(), name='post_list'),
    path('post/<int:pk>/', views.PostDetailView.as_view(), name='post_detail'),
    path('post/new/', views.PostCreateView.as_view(), name='post_create'),
    path('post/<int:pk>/edit/', views.PostUpdateView.as_view(), name='post_edit'),
    path('post/<int:pk>/delete/', views.PostDeleteView.as_view(), name='post_delete'),
]
```

### 5. Templates and Template Inheritance
```html
<!-- base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My Blog{% endblock %}</title>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <nav>
        <a href="{% url 'blog:post_list' %}">Home</a>
        {% if user.is_authenticated %}
            <a href="{% url 'blog:post_create' %}">New Post</a>
            <a href="{% url 'logout' %}">Logout</a>
        {% else %}
            <a href="{% url 'login' %}">Login</a>
        {% endif %}
    </nav>
    
    <main>
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }}">
                    {{ message }}
                </div>
            {% endfor %}
        {% endif %}
        
        {% block content %}{% endblock %}
    </main>
</body>
</html>

<!-- post_list.html -->
{% extends 'base.html' %}

{% block title %}Blog Posts{% endblock %}

{% block content %}
<h1>Blog Posts</h1>

{% for post in posts %}
    <article>
        <h2><a href="{% url 'blog:post_detail' post.pk %}">{{ post.title }}</a></h2>
        <p>By {{ post.author }} on {{ post.created_date|date:"F d, Y" }}</p>
        <p>{{ post.content|truncatewords:30 }}</p>
    </article>
{% empty %}
    <p>No posts yet.</p>
{% endfor %}

{% if is_paginated %}
    <div class="pagination">
        {% if page_obj.has_previous %}
            <a href="?page={{ page_obj.previous_page_number }}">Previous</a>
        {% endif %}
        
        <span class="current">
            Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}
        </span>
        
        {% if page_obj.has_next %}
            <a href="?page={{ page_obj.next_page_number }}">Next</a>
        {% endif %}
    </div>
{% endif %}
{% endblock %}
```

### 6. Forms and Validation
```python
# forms.py
from django import forms
from .models import Post, Comment

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content', 'published']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 10}),
            'published': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }
    
    def clean_title(self):
        title = self.cleaned_data.get('title')
        if len(title) < 5:
            raise forms.ValidationError("Title must be at least 5 characters long")
        return title

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 3})
        }
```

### 7. Admin Interface
```python
# admin.py
from django.contrib import admin
from .models import Post, Comment

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'created_date', 'published']
    list_filter = ['published', 'created_date', 'author']
    search_fields = ['title', 'content']
    prepopulated_fields = {'slug': ('title',)}
    raw_id_fields = ['author']
    date_hierarchy = 'created_date'
    ordering = ['-created_date']

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['post', 'author', 'created_date', 'approved']
    list_filter = ['approved', 'created_date']
    search_fields = ['content', 'author__username']
    actions = ['approve_comments']
    
    def approve_comments(self, request, queryset):
        queryset.update(approved=True)
    approve_comments.short_description = "Approve selected comments"
```

## Advanced Topics

### 1. Django REST Framework
```python
# serializers.py
from rest_framework import serializers
from .models import Post, Comment

class CommentSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = Comment
        fields = ['id', 'content', 'author', 'created_date']

class PostSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    comments = CommentSerializer(many=True, read_only=True)
    
    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'author', 'created_date', 'comments']

# views.py
from rest_framework import generics, permissions, filters
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from .models import Post, Comment
from .serializers import PostSerializer, CommentSerializer

class PostListCreateView(generics.ListCreateAPIView):
    queryset = Post.objects.filter(published=True)
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['title', 'content']
    ordering_fields = ['created_date', 'title']

class PostDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

@api_view(['GET', 'POST'])
@permission_classes([permissions.IsAuthenticated])
def comment_list(request, post_pk):
    if request.method == 'GET':
        comments = Comment.objects.filter(post_id=post_pk, approved=True)
        serializer = CommentSerializer(comments, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = CommentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user, post_id=post_pk)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
```

### 2. User Authentication and Permissions
```python
# views.py
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    form_class = PostForm
    template_name = 'blog/post_form.html'
    
    def test_func(self):
        post = self.get_object()
        return self.request.user == post.author

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('blog:post_list')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

# Custom permissions
class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.author == request.user
```

### 3. Database Migrations
```python
# Create migration
python manage.py makemigrations

# Apply migration
python manage.py migrate

# Custom migration
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('blog', '0001_initial'),
    ]
    
    operations = [
        migrations.AddField(
            model_name='post',
            name='slug',
            field=models.SlugField(max_length=200, unique=True),
        ),
        migrations.RunPython(
            code=populate_slugs,
            reverse_code=migrations.RunPython.noop,
        ),
    ]

def populate_slugs(apps, schema_editor):
    Post = apps.get_model('blog', 'Post')
    for post in Post.objects.all():
        post.slug = slugify(post.title)
        post.save()
```

### 4. Caching and Performance
```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}

# views.py
from django.core.cache import cache
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Cache for 15 minutes
def post_list(request):
    posts = Post.objects.filter(published=True)
    return render(request, 'blog/post_list.html', {'posts': posts})

# Template caching
{% load cache %}
{% cache 500 sidebar %}
    <div class="sidebar">
        <!-- sidebar content -->
    </div>
{% endcache %}
```

### 5. Testing
```python
# tests.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from .models import Post, Comment

class PostModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.post = Post.objects.create(
            title='Test Post',
            content='Test content',
            author=self.user
        )
    
    def test_post_creation(self):
        self.assertEqual(self.post.title, 'Test Post')
        self.assertEqual(self.post.author, self.user)
        self.assertFalse(self.post.published)
    
    def test_post_str_representation(self):
        self.assertEqual(str(self.post), 'Test Post')

class PostViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.post = Post.objects.create(
            title='Test Post',
            content='Test content',
            author=self.user,
            published=True
        )
    
    def test_post_list_view(self):
        response = self.client.get(reverse('blog:post_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Post')
    
    def test_post_detail_view(self):
        response = self.client.get(reverse('blog:post_detail', kwargs={'pk': self.post.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Post')
    
    def test_post_create_view_requires_login(self):
        response = self.client.get(reverse('blog:post_create'))
        self.assertRedirects(response, '/login/?next=/post/new/')
    
    def test_post_create_view_authenticated(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(reverse('blog:post_create'))
        self.assertEqual(response.status_code, 200)
```

## Best Practices

### 1. Project Structure
```
myproject/
├── manage.py
├── requirements.txt
├── .env
├── myproject/
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── __init__.py
│   ├── blog/
│   ├── users/
│   └── api/
├── static/
├── media/
└── templates/
```

### 2. Settings Management
```python
# settings/base.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get('SECRET_KEY')
DEBUG = False
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'apps.blog',
    'apps.users',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# settings/development.py
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# settings/production.py
from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT'),
    }
}
```

### 3. Security Best Practices
```python
# settings/production.py
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_SECONDS = 31536000
SECURE_REDIRECT_EXEMPT = []
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'

# Custom middleware for security
class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        return response
```

## Quick Checks

### Check 1: Model Relationships
```python
# What will this return?
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)

author = Author.objects.create(name="John Doe")
book = Book.objects.create(title="Python Guide", author=author)
print(book.author.name)  # What will this print?
```

### Check 2: URL Patterns
```python
# What URL will this match?
path('post/<int:year>/<int:month>/<int:day>/<slug:slug>/', views.post_detail, name='post_detail')
# URL: /post/2023/12/15/my-awesome-post/
```

### Check 3: Template Filters
```html
<!-- What will this display? -->
{{ post.created_date|date:"F d, Y" }}
<!-- If post.created_date is 2023-12-15 14:30:00 -->
```

## Lab Problems

### Lab 1: E-commerce Platform
Build a complete e-commerce platform with products, categories, shopping cart, and order management.

### Lab 2: Social Media Platform
Create a social media platform with user profiles, posts, comments, likes, and following system.

### Lab 3: Learning Management System
Develop an LMS with courses, lessons, assignments, and student progress tracking.

### Lab 4: Real Estate Platform
Build a real estate platform with property listings, search functionality, and agent management.

## AI Code Comparison
When working with AI-generated Django code, evaluate:
- **Model design** - are relationships and constraints properly defined?
- **Security implementation** - are CSRF tokens and permissions correctly handled?
- **Performance optimization** - are database queries optimized and caching implemented?
- **Code organization** - is the code properly structured following Django conventions?
- **Error handling** - are exceptions caught and handled appropriately?

## Next Steps
- Learn about Django's advanced features and extensions
- Master database optimization and query optimization
- Explore microservices architecture with Django
- Study deployment and scaling strategies
- Understand Django's security model and best practices
