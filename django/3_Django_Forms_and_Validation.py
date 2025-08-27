####################################################
## Django Framework - Lesson 3: Forms and Validation
####################################################

# Django provides a powerful form system for handling user input
# Forms handle validation, rendering, and data processing

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# 1. Basic Form Creation

class ContactForm(forms.Form):
    """Basic contact form"""
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your name'
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email'
        })
    )
    subject = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter subject'
        })
    )
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Enter your message'
        })
    )

# 2. Form Validation

class RegistrationForm(forms.Form):
    """Registration form with custom validation"""
    username = forms.CharField(
        max_length=30,
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9_]+$',
                message='Username can only contain letters, numbers, and underscores'
            )
        ],
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    password = forms.CharField(
        min_length=8,
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
    age = forms.IntegerField(
        validators=[MinValueValidator(13), MaxValueValidator(120)],
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    def clean(self):
        """Custom validation method"""
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords don't match")
        
        return cleaned_data
    
    def clean_username(self):
        """Custom field validation"""
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Username already exists")
        return username

# 3. Model Forms

from django.db import models

class Product(models.Model):
    """Product model for demonstration"""
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ProductForm(forms.ModelForm):
    """Model form for Product"""
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'category']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'price': forms.NumberInput(attrs={'class': 'form-control'}),
            'category': forms.Select(attrs={'class': 'form-control'})
        }
        labels = {
            'name': 'Product Name',
            'description': 'Product Description',
            'price': 'Price ($)',
            'category': 'Category'
        }
        help_texts = {
            'name': 'Enter the product name',
            'description': 'Provide a detailed description',
            'price': 'Enter the price in dollars',
            'category': 'Select the product category'
        }

# 4. Custom Form Widgets

class DateInput(forms.DateInput):
    """Custom date input widget"""
    input_type = 'date'

class TimeInput(forms.TimeInput):
    """Custom time input widget"""
    input_type = 'time'

class EventForm(forms.Form):
    """Event form with custom widgets"""
    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    date = forms.DateField(
        widget=DateInput(attrs={'class': 'form-control'})
    )
    time = forms.TimeField(
        widget=TimeInput(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4})
    )

# 5. Form Sets

from django.forms import formset_factory

class OrderItemForm(forms.Form):
    """Form for individual order items"""
    product = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Product name'})
    )
    quantity = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    price = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

# Create a formset
OrderItemFormSet = formset_factory(
    OrderItemForm,
    extra=3,  # Number of empty forms to display
    can_delete=True,  # Allow deletion of forms
    can_order=True    # Allow reordering of forms
)

# 6. Advanced Form Features

class SearchForm(forms.Form):
    """Advanced search form with multiple field types"""
    query = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search...'
        })
    )
    category = forms.ChoiceField(
        choices=[
            ('', 'All Categories'),
            ('electronics', 'Electronics'),
            ('clothing', 'Clothing'),
            ('books', 'Books'),
            ('sports', 'Sports')
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    min_price = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Min price'
        })
    )
    max_price = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Max price'
        })
    )
    sort_by = forms.ChoiceField(
        choices=[
            ('name', 'Name'),
            ('price', 'Price'),
            ('date', 'Date Added')
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    in_stock = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# 7. File Upload Forms

class FileUploadForm(forms.Form):
    """Form for file uploads"""
    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3})
    )
    file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    
    def clean_file(self):
        """Validate uploaded file"""
        file = self.cleaned_data.get('file')
        if file:
            # Check file size (5MB limit)
            if file.size > 5 * 1024 * 1024:
                raise forms.ValidationError("File size must be under 5MB")
            
            # Check file extension
            allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
            import os
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in allowed_extensions:
                raise forms.ValidationError("Only PDF, DOC, DOCX, and TXT files are allowed")
        
        return file

# 8. Custom Form Fields

class PhoneNumberField(forms.CharField):
    """Custom phone number field with validation"""
    def __init__(self, *args, **kwargs):
        kwargs['max_length'] = 15
        kwargs['widget'] = forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '(123) 456-7890'
        })
        super().__init__(*args, **kwargs)
    
    def clean(self, value):
        """Clean and validate phone number"""
        value = super().clean(value)
        if value:
            # Remove all non-digit characters
            digits = ''.join(filter(str.isdigit, value))
            if len(digits) != 10:
                raise forms.ValidationError("Phone number must have 10 digits")
        return value

class CustomerForm(forms.Form):
    """Customer form with custom phone field"""
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    phone = PhoneNumberField()
    address = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3})
    )

# 9. Form Processing in Views

# Example view for processing forms
"""
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import ContactForm, ProductForm, SearchForm

def contact_view(request):
    '''Process contact form'''
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Process the form data
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            
            # Save to database or send email
            # ... processing logic ...
            
            messages.success(request, 'Your message has been sent successfully!')
            return redirect('contact_success')
    else:
        form = ContactForm()
    
    return render(request, 'contact.html', {'form': form})

def product_create_view(request):
    '''Create new product'''
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            product = form.save()
            messages.success(request, f'Product "{product.name}" created successfully!')
            return redirect('product_detail', pk=product.pk)
    else:
        form = ProductForm()
    
    return render(request, 'product_form.html', {'form': form})

def search_view(request):
    '''Search products'''
    form = SearchForm(request.GET)
    products = []
    
    if form.is_valid():
        query = form.cleaned_data.get('query')
        category = form.cleaned_data.get('category')
        min_price = form.cleaned_data.get('min_price')
        max_price = form.cleaned_data.get('max_price')
        sort_by = form.cleaned_data.get('sort_by')
        in_stock = form.cleaned_data.get('in_stock')
        
        # Build query based on form data
        products = Product.objects.filter(is_active=True)
        
        if query:
            products = products.filter(name__icontains=query)
        if category:
            products = products.filter(category=category)
        if min_price:
            products = products.filter(price__gte=min_price)
        if max_price:
            products = products.filter(price__lte=max_price)
        if in_stock:
            products = products.filter(stock__gt=0)
        
        if sort_by:
            products = products.order_by(sort_by)
    
    return render(request, 'search.html', {
        'form': form,
        'products': products
    })
"""

# 10. Form Templates

# Example template for rendering forms
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
    
    {% if form.non_field_errors %}
        <div class="alert alert-danger">
            {% for error in form.non_field_errors %}
                {{ error }}
            {% endfor %}
        </div>
    {% endif %}
    
    {% for field in form %}
        <div class="form-group">
            {{ field.label_tag }}
            {{ field }}
            {% if field.help_text %}
                <small class="form-text text-muted">{{ field.help_text }}</small>
            {% endif %}
            {% if field.errors %}
                <div class="alert alert-danger">
                    {% for error in field.errors %}
                        {{ error }}
                    {% endfor %}
                </div>
            {% endif %}
        </div>
    {% endfor %}
    
    <button type="submit" class="btn btn-primary">Submit</button>
</form>
{% endblock %}
"""

# 11. AJAX Form Handling

# Example JavaScript for AJAX form submission
"""
<script>
document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    fetch('/contact/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Message sent successfully!');
            this.reset();
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred');
    });
});
</script>
"""

# 12. Form Security

# CSRF Protection (automatically included in Django forms)
"""
# In settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF protection
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
"""

# 13. Exercises

# Exercise 1: Create a User Registration Form
"""
1. Create a UserRegistrationForm with username, email, password, and confirm_password
2. Add custom validation for password strength
3. Check if username and email already exist
4. Create a view to handle the registration
5. Create a template to display the form
"""

# Exercise 2: Create a Product Review Form
"""
1. Create a Review model with rating, comment, and user fields
2. Create a ReviewForm with custom validation
3. Ensure rating is between 1-5
4. Add a view to handle review submission
5. Display reviews on product detail page
"""

# Exercise 3: Create a Multi-Step Form
"""
1. Create a multi-step registration form
2. Use sessions to store form data between steps
3. Validate each step before proceeding
4. Allow users to go back and edit previous steps
5. Submit all data at the final step
"""

# Exercise 4: Create a Dynamic Form
"""
1. Create a form that changes based on user selections
2. Use JavaScript to show/hide form fields
3. Validate dynamic fields on the server side
4. Handle form submission with dynamic data
"""

# 14. Best Practices

"""
1. Always validate data on both client and server side
2. Use ModelForms when working with models
3. Provide clear error messages to users
4. Use appropriate field types and widgets
5. Implement CSRF protection for all forms
6. Sanitize user input to prevent XSS attacks
7. Use form sets for multiple related forms
8. Provide helpful help_text for form fields
9. Use custom validators for complex validation
10. Test forms thoroughly with various inputs
"""

# Next Steps:
# - Learn about Django Class-Based Views
# - Understand Django REST Framework for APIs
# - Master Django Templates and template inheritance
# - Explore Django signals and middleware
# - Learn about Django caching and optimization 