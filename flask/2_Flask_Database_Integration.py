####################################################
## Flask Framework - Lesson 2: Database Integration
####################################################

# Flask-SQLAlchemy provides a simple way to use SQLAlchemy with Flask
# It handles database connections, models, and migrations

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import os

# 1. Basic Setup

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 2. Basic Models

class User(db.Model):
    """User model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    posts = db.relationship('Post', backref='author', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    """Blog post model"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_published = db.Column(db.Boolean, default=False)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
    
    # Relationships
    comments = db.relationship('Comment', backref='post', lazy=True)
    tags = db.relationship('Tag', secondary='post_tags', lazy='subquery',
                          backref=db.backref('posts', lazy=True))
    
    def __repr__(self):
        return f'<Post {self.title}>'

class Category(db.Model):
    """Category model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    
    # Relationships
    posts = db.relationship('Post', backref='category', lazy=True)
    
    def __repr__(self):
        return f'<Category {self.name}>'

class Comment(db.Model):
    """Comment model"""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_approved = db.Column(db.Boolean, default=False)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    
    def __repr__(self):
        return f'<Comment {self.id}>'

class Tag(db.Model):
    """Tag model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    
    def __repr__(self):
        return f'<Tag {self.name}>'

# Association table for many-to-many relationship
post_tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

# 3. Advanced Models

class Product(db.Model):
    """E-commerce product model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    stock = db.Column(db.Integer, default=0)
    sku = db.Column(db.String(50), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    order_items = db.relationship('OrderItem', backref='product', lazy=True)
    
    def __repr__(self):
        return f'<Product {self.name}>'
    
    @property
    def is_in_stock(self):
        return self.stock > 0

class Order(db.Model):
    """Order model"""
    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(20), unique=True, nullable=False)
    total_amount = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    items = db.relationship('OrderItem', backref='order', lazy=True)
    
    def __repr__(self):
        return f'<Order {self.order_number}>'

class OrderItem(db.Model):
    """Order item model"""
    id = db.Column(db.Integer, primary_key=True)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Foreign keys
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    
    def __repr__(self):
        return f'<OrderItem {self.id}>'
    
    @property
    def total_price(self):
        return self.quantity * self.price

# 4. Database Operations

# Create all tables
def init_db():
    """Initialize the database"""
    with app.app_context():
        db.create_all()
        print("Database tables created!")

# Seed data
def seed_data():
    """Add sample data to the database"""
    with app.app_context():
        # Create categories
        categories = [
            Category(name='Technology', description='Tech-related posts'),
            Category(name='Travel', description='Travel experiences'),
            Category(name='Food', description='Food and recipes'),
            Category(name='Lifestyle', description='Lifestyle tips')
        ]
        
        for category in categories:
            db.session.add(category)
        
        # Create tags
        tags = [
            Tag(name='python'),
            Tag(name='flask'),
            Tag(name='web-development'),
            Tag(name='tutorial')
        ]
        
        for tag in tags:
            db.session.add(tag)
        
        db.session.commit()
        print("Sample data added!")

# 5. Basic CRUD Operations

@app.route('/users')
def list_users():
    """List all users"""
    users = User.query.all()
    return render_template('users/list.html', users=users)

@app.route('/users/<int:user_id>')
def get_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return render_template('users/detail.html', user=user)

@app.route('/users/create', methods=['GET', 'POST'])
def create_user():
    """Create a new user"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return redirect(url_for('create_user'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'error')
            return redirect(url_for('create_user'))
        
        # Create new user (in production, hash the password)
        user = User(username=username, email=email, password_hash=password)
        db.session.add(user)
        db.session.commit()
        
        flash('User created successfully!', 'success')
        return redirect(url_for('list_users'))
    
    return render_template('users/create.html')

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
def edit_user(user_id):
    """Edit a user"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.is_active = 'is_active' in request.form
        
        db.session.commit()
        flash('User updated successfully!', 'success')
        return redirect(url_for('get_user', user_id=user.id))
    
    return render_template('users/edit.html', user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    
    flash('User deleted successfully!', 'success')
    return redirect(url_for('list_users'))

# 6. Advanced Queries

@app.route('/posts')
def list_posts():
    """List all posts with filtering and pagination"""
    page = request.args.get('page', 1, type=int)
    category_id = request.args.get('category', type=int)
    search = request.args.get('search', '')
    
    # Build query
    query = Post.query
    
    if category_id:
        query = query.filter_by(category_id=category_id)
    
    if search:
        query = query.filter(Post.title.contains(search) | Post.content.contains(search))
    
    # Only show published posts
    query = query.filter_by(is_published=True)
    
    # Order by creation date
    query = query.order_by(Post.created_at.desc())
    
    # Paginate results
    posts = query.paginate(page=page, per_page=10, error_out=False)
    
    # Get categories for filter
    categories = Category.query.all()
    
    return render_template('posts/list.html', posts=posts, categories=categories)

@app.route('/posts/<int:post_id>')
def get_post(post_id):
    """Get a specific post with comments"""
    post = Post.query.get_or_404(post_id)
    comments = Comment.query.filter_by(post_id=post_id, is_approved=True).order_by(Comment.created_at.desc()).all()
    
    return render_template('posts/detail.html', post=post, comments=comments)

# 7. Complex Relationships

@app.route('/posts/<int:post_id>/comments', methods=['POST'])
def add_comment(post_id):
    """Add a comment to a post"""
    post = Post.query.get_or_404(post_id)
    
    content = request.form.get('content')
    if not content:
        flash('Comment content is required!', 'error')
        return redirect(url_for('get_post', post_id=post_id))
    
    # In a real app, get user from session
    user = User.query.first()  # For demo purposes
    
    comment = Comment(
        content=content,
        user_id=user.id,
        post_id=post_id
    )
    
    db.session.add(comment)
    db.session.commit()
    
    flash('Comment added successfully!', 'success')
    return redirect(url_for('get_post', post_id=post_id))

# 8. API Endpoints

@app.route('/api/posts')
def api_posts():
    """API endpoint for posts"""
    posts = Post.query.filter_by(is_published=True).order_by(Post.created_at.desc()).all()
    
    return jsonify([{
        'id': post.id,
        'title': post.title,
        'content': post.content[:200] + '...' if len(post.content) > 200 else post.content,
        'author': post.author.username,
        'created_at': post.created_at.isoformat(),
        'category': post.category.name if post.category else None
    } for post in posts])

@app.route('/api/posts/<int:post_id>')
def api_post(post_id):
    """API endpoint for a specific post"""
    post = Post.query.get_or_404(post_id)
    
    return jsonify({
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author.username,
        'created_at': post.created_at.isoformat(),
        'category': post.category.name if post.category else None,
        'tags': [tag.name for tag in post.tags]
    })

@app.route('/api/posts', methods=['POST'])
def api_create_post():
    """API endpoint to create a post"""
    data = request.get_json()
    
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({'error': 'Title and content are required'}), 400
    
    # In a real app, get user from authentication
    user = User.query.first()  # For demo purposes
    
    post = Post(
        title=data['title'],
        content=data['content'],
        user_id=user.id,
        is_published=data.get('is_published', False)
    )
    
    db.session.add(post)
    db.session.commit()
    
    return jsonify({
        'id': post.id,
        'title': post.title,
        'message': 'Post created successfully'
    }), 201

# 9. Database Migrations

# Create migration
"""
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
"""

# 10. Database Events and Hooks

class Post(db.Model):
    # ... existing fields ...
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def before_insert(self):
        """Hook before inserting"""
        if not self.title:
            raise ValueError("Title is required")
    
    def after_insert(self):
        """Hook after inserting"""
        print(f"New post created: {self.title}")

# 11. Database Utilities

def get_user_stats(user_id):
    """Get user statistics"""
    user = User.query.get_or_404(user_id)
    
    stats = {
        'total_posts': Post.query.filter_by(user_id=user_id).count(),
        'published_posts': Post.query.filter_by(user_id=user_id, is_published=True).count(),
        'total_comments': Comment.query.filter_by(user_id=user_id).count(),
        'approved_comments': Comment.query.filter_by(user_id=user_id, is_approved=True).count(),
        'member_since': user.created_at.strftime('%B %Y')
    }
    
    return stats

def search_posts(query, limit=10):
    """Search posts by title or content"""
    return Post.query.filter(
        db.or_(
            Post.title.contains(query),
            Post.content.contains(query)
        )
    ).filter_by(is_published=True).limit(limit).all()

def get_popular_tags(limit=10):
    """Get most popular tags"""
    return db.session.query(Tag, db.func.count(post_tags.c.post_id).label('count'))\
        .join(post_tags)\
        .group_by(Tag.id)\
        .order_by(db.desc('count'))\
        .limit(limit)\
        .all()

# 12. Error Handling

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

# 13. Exercises

# Exercise 1: Create a Blog System
"""
1. Create User, Post, Category, and Comment models
2. Implement CRUD operations for posts
3. Add user authentication
4. Create a comment system
5. Add categories and tags
"""

# Exercise 2: Create an E-commerce System
"""
1. Create Product, Order, and OrderItem models
2. Implement shopping cart functionality
3. Add order processing
4. Create inventory management
5. Add payment integration
"""

# Exercise 3: Create a Social Media Platform
"""
1. Create User, Post, Comment, and Like models
2. Implement user following system
3. Add post sharing functionality
4. Create notification system
5. Add real-time features
"""

# Exercise 4: Create an API
"""
1. Create RESTful API endpoints
2. Add authentication and authorization
3. Implement rate limiting
4. Add API documentation
5. Create API testing
"""

# 14. Best Practices

"""
1. Always use migrations for database changes
2. Use appropriate data types and constraints
3. Implement proper relationships between models
4. Add indexes for frequently queried fields
5. Use transactions for complex operations
6. Implement proper error handling
7. Use database events for data integrity
8. Optimize queries for performance
9. Use connection pooling in production
10. Regular database backups
"""

# 15. Production Considerations

# Database configuration for production
"""
# PostgreSQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/dbname'

# MySQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://user:password@localhost/dbname'

# Environment variables
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
"""

# Connection pooling
"""
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:password@localhost/dbname',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30
)
"""

# Next Steps:
# - Learn about Flask-Login for authentication
# - Understand Flask-WTF for form handling
# - Master Flask blueprints and application structure
# - Explore Flask extensions and ecosystem
# - Learn about Flask testing and deployment 