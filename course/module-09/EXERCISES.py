"""
Module 9: Web Development with Flask - Exercises
Complete these exercises to master Flask web development.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Exercise 1: Basic Flask Application
def create_basic_app():
    """Create a basic Flask application with routing."""
    app = Flask(__name__)
    app.secret_key = 'your-secret-key-here'
    
    @app.route('/')
    def home():
        """Home page route."""
        return render_template('index.html', 
                             title='Home', 
                             current_time=datetime.now())
    
    @app.route('/about')
    def about():
        """About page route."""
        return render_template('about.html', title='About')
    
    @app.route('/contact', methods=['GET', 'POST'])
    def contact():
        """Contact page with form handling."""
        if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email')
            message = request.form.get('message')
            
            # Process form data
            flash(f'Thank you {name}! Your message has been sent.', 'success')
            return redirect(url_for('contact'))
        
        return render_template('contact.html', title='Contact')
    
    @app.route('/api/status')
    def api_status():
        """API endpoint for status check."""
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    return app

# Exercise 2: User Authentication System
class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Set password hash."""
        # In a real app, use proper password hashing
        self.password_hash = password
    
    def check_password(self, password):
        """Check password."""
        return self.password_hash == password

class LoginForm(FlaskForm):
    """Login form."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    """Registration form."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Register')

def create_auth_app():
    """Create Flask app with authentication."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///auth_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    @app.route('/')
    def home():
        return render_template('index.html', user=current_user)
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and user.check_password(form.password.data):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        
        return render_template('login.html', form=form)
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = RegistrationForm()
        if form.validate_on_submit():
            user = User(
                username=form.username.data,
                email=form.email.data
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html', form=form)
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template('dashboard.html', user=current_user)
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out', 'info')
        return redirect(url_for('home'))
    
    return app

# Exercise 3: RESTful API
class Post(db.Model):
    """Post model for blog API."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'author_id': self.author_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

def create_api_app():
    """Create Flask app with RESTful API."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///api_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    @app.route('/api/posts', methods=['GET'])
    @limiter.limit("10 per minute")
    def get_posts():
        """Get all posts."""
        posts = Post.query.all()
        return jsonify([post.to_dict() for post in posts])
    
    @app.route('/api/posts/<int:post_id>', methods=['GET'])
    def get_post(post_id):
        """Get a specific post."""
        post = Post.query.get_or_404(post_id)
        return jsonify(post.to_dict())
    
    @app.route('/api/posts', methods=['POST'])
    def create_post():
        """Create a new post."""
        data = request.get_json()
        
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        post = Post(
            title=data['title'],
            content=data['content'],
            author_id=data.get('author_id', 1)  # Default author
        )
        
        db.session.add(post)
        db.session.commit()
        
        return jsonify(post.to_dict()), 201
    
    @app.route('/api/posts/<int:post_id>', methods=['PUT'])
    def update_post(post_id):
        """Update a post."""
        post = Post.query.get_or_404(post_id)
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        post.title = data.get('title', post.title)
        post.content = data.get('content', post.content)
        post.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify(post.to_dict())
    
    @app.route('/api/posts/<int:post_id>', methods=['DELETE'])
    def delete_post(post_id):
        """Delete a post."""
        post = Post.query.get_or_404(post_id)
        db.session.delete(post)
        db.session.commit()
        
        return jsonify({'message': 'Post deleted successfully'})
    
    return app

# Exercise 4: E-commerce Application
class Product(db.Model):
    """Product model for e-commerce."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    stock_quantity = db.Column(db.Integer, default=0)
    category = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': self.price,
            'stock_quantity': self.stock_quantity,
            'category': self.category,
            'created_at': self.created_at.isoformat()
        }

class CartItem(db.Model):
    """Shopping cart item model."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    product = db.relationship('Product', backref='cart_items')

def create_ecommerce_app():
    """Create e-commerce Flask app."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecommerce.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    @app.route('/')
    def home():
        """Home page with featured products."""
        products = Product.query.limit(6).all()
        return render_template('ecommerce/index.html', products=products)
    
    @app.route('/products')
    def products():
        """Product listing page."""
        category = request.args.get('category')
        page = request.args.get('page', 1, type=int)
        
        query = Product.query
        if category:
            query = query.filter_by(category=category)
        
        products = query.paginate(
            page=page, per_page=12, error_out=False
        )
        
        return render_template('ecommerce/products.html', 
                             products=products, 
                             category=category)
    
    @app.route('/product/<int:product_id>')
    def product_detail(product_id):
        """Product detail page."""
        product = Product.query.get_or_404(product_id)
        return render_template('ecommerce/product_detail.html', product=product)
    
    @app.route('/cart')
    @login_required
    def cart():
        """Shopping cart page."""
        cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
        total = sum(item.product.price * item.quantity for item in cart_items)
        return render_template('ecommerce/cart.html', 
                             cart_items=cart_items, 
                             total=total)
    
    @app.route('/add_to_cart/<int:product_id>', methods=['POST'])
    @login_required
    def add_to_cart(product_id):
        """Add product to cart."""
        product = Product.query.get_or_404(product_id)
        quantity = request.form.get('quantity', 1, type=int)
        
        # Check if item already in cart
        cart_item = CartItem.query.filter_by(
            user_id=current_user.id, 
            product_id=product_id
        ).first()
        
        if cart_item:
            cart_item.quantity += quantity
        else:
            cart_item = CartItem(
                user_id=current_user.id,
                product_id=product_id,
                quantity=quantity
            )
            db.session.add(cart_item)
        
        db.session.commit()
        flash(f'{product.name} added to cart!', 'success')
        return redirect(url_for('product_detail', product_id=product_id))
    
    @app.route('/remove_from_cart/<int:item_id>', methods=['POST'])
    @login_required
    def remove_from_cart(item_id):
        """Remove item from cart."""
        cart_item = CartItem.query.get_or_404(item_id)
        if cart_item.user_id == current_user.id:
            db.session.delete(cart_item)
            db.session.commit()
            flash('Item removed from cart', 'info')
        
        return redirect(url_for('cart'))
    
    return app

# Exercise 5: Real-time Chat Application
from flask_socketio import SocketIO, emit, join_room, leave_room

class Message(db.Model):
    """Message model for chat."""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    room = db.Column(db.String(100), default='general')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='messages')

def create_chat_app():
    """Create real-time chat Flask app."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @app.route('/')
    def index():
        """Chat home page."""
        return render_template('chat/index.html')
    
    @app.route('/chat/<room>')
    @login_required
    def chat_room(room):
        """Chat room page."""
        messages = Message.query.filter_by(room=room).order_by(Message.timestamp).all()
        return render_template('chat/room.html', room=room, messages=messages)
    
    @socketio.on('join')
    def on_join(data):
        """Handle user joining a room."""
        room = data['room']
        join_room(room)
        emit('status', {'msg': f'{current_user.username} joined the room'}, room=room)
    
    @socketio.on('leave')
    def on_leave(data):
        """Handle user leaving a room."""
        room = data['room']
        leave_room(room)
        emit('status', {'msg': f'{current_user.username} left the room'}, room=room)
    
    @socketio.on('message')
    def handle_message(data):
        """Handle incoming messages."""
        room = data['room']
        content = data['message']
        
        # Save message to database
        message = Message(
            content=content,
            user_id=current_user.id,
            room=room
        )
        db.session.add(message)
        db.session.commit()
        
        # Emit message to room
        emit('message', {
            'content': content,
            'username': current_user.username,
            'timestamp': message.timestamp.isoformat()
        }, room=room)
    
    return app

# Exercise 6: File Upload and Management
from werkzeug.utils import secure_filename
import uuid

class FileUpload(db.Model):
    """File upload model."""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='uploads')

def create_file_upload_app():
    """Create Flask app with file upload functionality."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///file_upload.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Initialize extensions
    db.init_app(app)
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
    
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def home():
        """Home page with file upload form."""
        return render_template('file_upload/index.html')
    
    @app.route('/upload', methods=['POST'])
    @login_required
    def upload_file():
        """Handle file upload."""
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save file
            file.save(file_path)
            
            # Save file info to database
            file_upload = FileUpload(
                filename=unique_filename,
                original_filename=filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                file_type=file.content_type,
                user_id=current_user.id
            )
            db.session.add(file_upload)
            db.session.commit()
            
            flash('File uploaded successfully!', 'success')
        else:
            flash('Invalid file type', 'error')
        
        return redirect(url_for('home'))
    
    @app.route('/files')
    @login_required
    def list_files():
        """List user's uploaded files."""
        files = FileUpload.query.filter_by(user_id=current_user.id).all()
        return render_template('file_upload/files.html', files=files)
    
    @app.route('/download/<int:file_id>')
    @login_required
    def download_file(file_id):
        """Download a file."""
        file_upload = FileUpload.query.get_or_404(file_id)
        if file_upload.user_id != current_user.id:
            flash('Access denied', 'error')
            return redirect(url_for('list_files'))
        
        return send_file(file_upload.file_path, as_attachment=True)
    
    return app

# Exercise 7: API Testing and Documentation
def create_api_documentation():
    """Create API documentation."""
    api_docs = {
        "title": "Flask API Documentation",
        "version": "1.0.0",
        "description": "A comprehensive Flask API with authentication and CRUD operations",
        "endpoints": {
            "authentication": {
                "POST /api/auth/login": {
                    "description": "User login",
                    "parameters": {
                        "username": "string (required)",
                        "password": "string (required)"
                    },
                    "response": {
                        "200": "Login successful",
                        "401": "Invalid credentials"
                    }
                },
                "POST /api/auth/register": {
                    "description": "User registration",
                    "parameters": {
                        "username": "string (required)",
                        "email": "string (required)",
                        "password": "string (required)"
                    },
                    "response": {
                        "201": "User created successfully",
                        "400": "Validation error"
                    }
                }
            },
            "posts": {
                "GET /api/posts": {
                    "description": "Get all posts",
                    "response": "List of posts"
                },
                "POST /api/posts": {
                    "description": "Create a new post",
                    "parameters": {
                        "title": "string (required)",
                        "content": "string (required)"
                    },
                    "response": "Created post object"
                }
            }
        }
    }
    return api_docs

# Test Functions
def test_exercises():
    """Test all Flask exercises."""
    print("Testing Module 9 Flask Exercises...")
    
    # Test 1: Basic App
    print("\n1. Testing Basic Flask App:")
    basic_app = create_basic_app()
    print("Basic Flask app created successfully")
    
    # Test 2: Authentication App
    print("\n2. Testing Authentication App:")
    auth_app = create_auth_app()
    print("Authentication app created successfully")
    
    # Test 3: API App
    print("\n3. Testing API App:")
    api_app = create_api_app()
    print("API app created successfully")
    
    # Test 4: E-commerce App
    print("\n4. Testing E-commerce App:")
    ecommerce_app = create_ecommerce_app()
    print("E-commerce app created successfully")
    
    # Test 5: Chat App
    print("\n5. Testing Chat App:")
    chat_app = create_chat_app()
    print("Chat app created successfully")
    
    # Test 6: File Upload App
    print("\n6. Testing File Upload App:")
    file_app = create_file_upload_app()
    print("File upload app created successfully")
    
    # Test 7: API Documentation
    print("\n7. Testing API Documentation:")
    docs = create_api_documentation()
    print(f"API documentation created with {len(docs['endpoints'])} endpoint categories")
    
    print("\nAll Flask exercises completed!")

if __name__ == "__main__":
    test_exercises()
