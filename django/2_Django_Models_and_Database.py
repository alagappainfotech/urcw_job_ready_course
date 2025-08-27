####################################################
## Django Framework - Lesson 2: Models and Database
####################################################

# Django's ORM (Object-Relational Mapping) allows you to interact with databases
# using Python objects instead of writing raw SQL.

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

# 1. Basic Model Definition

class Book(models.Model):
    """A simple book model"""
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_date = models.DateField()
    isbn = models.CharField(max_length=13, unique=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    is_available = models.BooleanField(default=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['title']
        verbose_name_plural = "Books"

# 2. Field Types

class Product(models.Model):
    """Example of different field types"""
    
    # Text fields
    name = models.CharField(max_length=100)  # Short text
    description = models.TextField()  # Long text
    slug = models.SlugField(unique=True)  # URL-friendly text
    
    # Numeric fields
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()
    rating = models.FloatField()
    
    # Date and time fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expiry_date = models.DateField()
    
    # File fields
    image = models.ImageField(upload_to='products/', blank=True)
    document = models.FileField(upload_to='documents/', blank=True)
    
    # Choice fields
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('draft', 'Draft'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    
    # Boolean field
    is_featured = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name

# 3. Model Relationships

class Category(models.Model):
    """Category model for products"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class Product(models.Model):
    """Product model with relationships"""
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return self.name

class Order(models.Model):
    """Order model with many-to-many relationship"""
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    products = models.ManyToManyField(Product, through='OrderItem')
    order_date = models.DateTimeField(auto_now_add=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"Order {self.id} by {self.customer.username}"

class OrderItem(models.Model):
    """Intermediate model for Order-Product relationship"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"{self.quantity}x {self.product.name}"

# 4. Advanced Model Features

class BlogPost(models.Model):
    """Blog post with advanced features"""
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)
    is_published = models.BooleanField(default=False)
    
    # Custom manager
    objects = models.Manager()
    published = models.Manager()  # We'll define this in the manager
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if self.is_published and not self.published_at:
            self.published_at = timezone.now()
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Blog Post"
        verbose_name_plural = "Blog Posts"

# 5. Custom Model Manager

class PublishedManager(models.Manager):
    """Custom manager for published blog posts"""
    
    def get_queryset(self):
        return super().get_queryset().filter(is_published=True, published_at__lte=timezone.now())

class BlogPost(models.Model):
    # ... other fields ...
    
    objects = models.Manager()
    published = PublishedManager()

# 6. Model Methods and Properties

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_year = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    @property
    def is_old(self):
        """Check if the book is older than 10 years"""
        return timezone.now().year - self.publication_year > 10
    
    def get_discount_price(self, discount_percent=10):
        """Calculate discounted price"""
        discount = self.price * (discount_percent / 100)
        return self.price - discount
    
    def __str__(self):
        return f"{self.title} by {self.author}"

# 7. Model Validation

class Student(models.Model):
    """Student model with custom validation"""
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    age = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(120)])
    grade = models.DecimalField(max_digits=3, decimal_places=2, validators=[MinValueValidator(0), MaxValueValidator(4)])
    
    def clean(self):
        """Custom validation"""
        from django.core.exceptions import ValidationError
        
        if self.age < 5:
            raise ValidationError("Student must be at least 5 years old")
        
        if self.grade < 0 or self.grade > 4:
            raise ValidationError("Grade must be between 0 and 4")
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

# 8. Database Queries

# Basic queries (to be used in views or shell)
"""
# Get all books
books = Book.objects.all()

# Get a specific book
book = Book.objects.get(id=1)
book = Book.objects.get(title="Python Programming")

# Filter books
python_books = Book.objects.filter(title__icontains="Python")
expensive_books = Book.objects.filter(price__gte=50.00)
recent_books = Book.objects.filter(publication_year__gte=2020)

# Exclude books
non_python_books = Book.objects.exclude(title__icontains="Python")

# Order books
books_by_price = Book.objects.order_by('price')
books_by_price_desc = Book.objects.order_by('-price')

# Limit results
first_five_books = Book.objects.all()[:5]

# Complex queries
from django.db.models import Q
python_or_java_books = Book.objects.filter(
    Q(title__icontains="Python") | Q(title__icontains="Java")
)

# Aggregation
from django.db.models import Avg, Count, Max, Min, Sum
avg_price = Book.objects.aggregate(Avg('price'))
book_count = Book.objects.count()
max_price = Book.objects.aggregate(Max('price'))
"""

# 9. Related Object Queries

"""
# Get all products in a category
category = Category.objects.get(name="Electronics")
products = category.products.all()

# Get category for a product
product = Product.objects.get(name="Laptop")
category = product.category

# Get all orders for a user
user = User.objects.get(username="john")
orders = user.order_set.all()

# Get all products in an order
order = Order.objects.get(id=1)
products = order.products.all()

# Get order items with quantities
order_items = order.orderitem_set.all()
"""

# 10. Database Migrations

# Creating migrations:
"""
python manage.py makemigrations
python manage.py makemigrations myapp
python manage.py makemigrations --name add_author_field
"""

# Applying migrations:
"""
python manage.py migrate
python manage.py migrate myapp
python manage.py migrate myapp 0001
"""

# Migration commands:
"""
python manage.py showmigrations          # Show migration status
python manage.py sqlmigrate myapp 0001  # Show SQL for migration
python manage.py migrate --plan          # Show migration plan
"""

# 11. Practical Example: E-commerce Models

class Customer(models.Model):
    """Customer model"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, blank=True)
    address = models.TextField(blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    
    def __str__(self):
        return self.user.username

class Category(models.Model):
    """Product category"""
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return self.name

class Product(models.Model):
    """Product model"""
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    @property
    def is_in_stock(self):
        return self.stock > 0

class Order(models.Model):
    """Order model"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Order {self.id} - {self.customer.user.username}"

class OrderItem(models.Model):
    """Order item model"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"{self.quantity}x {self.product.name}"
    
    @property
    def total_price(self):
        return self.quantity * self.price

# 12. Exercises

# Exercise 1: Create a Library Management System
"""
Create models for:
- Book (title, author, isbn, publication_year, genre)
- Member (name, email, phone, membership_date)
- Borrowing (book, member, borrow_date, return_date, is_returned)
"""

# Exercise 2: Create a Blog System with Comments
"""
Create models for:
- Post (title, content, author, created_at, updated_at)
- Comment (post, author, content, created_at, is_approved)
- Tag (name, description)
- PostTag (intermediate model for many-to-many)
"""

# Exercise 3: Create a Restaurant Menu System
"""
Create models for:
- Category (name, description)
- MenuItem (name, description, price, category, is_available)
- Order (customer_name, table_number, total_amount, order_date)
- OrderItem (order, menu_item, quantity, price)
"""

# Exercise 4: Add Advanced Features
"""
1. Add custom model methods
2. Add model validation
3. Add custom managers
4. Add model signals
5. Add model admin configuration
"""

# 13. Best Practices

"""
1. Use meaningful field names
2. Set appropriate field constraints
3. Use related_name for reverse relationships
4. Add __str__ methods to all models
5. Use Meta classes for ordering and verbose names
6. Validate data in model clean() method
7. Use database indexes for frequently queried fields
8. Keep models focused and single-purpose
9. Use migrations for all database changes
10. Write tests for your models
"""

# 14. Common Field Options

"""
null=True              # Allow NULL in database
blank=True             # Allow empty in forms
default=value          # Set default value
unique=True            # Ensure uniqueness
db_index=True          # Create database index
choices=CHOICES        # Limit to specific choices
help_text="text"       # Help text for forms
verbose_name="text"    # Human-readable name
"""

# Next Steps:
# - Learn about Django Forms and ModelForms
# - Understand Django Admin interface
# - Master database queries and optimization
# - Learn about Django signals
# - Explore database transactions and atomic operations 