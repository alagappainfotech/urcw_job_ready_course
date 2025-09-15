# Module 12: DevOps and Deployment - Complete Guide

## Learning Objectives
By the end of this module, you will be able to:
- Master containerization with Docker and container orchestration
- Implement CI/CD pipelines for automated testing and deployment
- Deploy applications to cloud platforms (AWS, GCP, Azure)
- Configure monitoring, logging, and alerting systems
- Manage infrastructure as code using Terraform and Ansible
- Implement security best practices in deployment
- Scale applications for high availability and performance

## Core Concepts

### 1. Containerization with Docker
Docker enables packaging applications and their dependencies into lightweight, portable containers.

```dockerfile
# Dockerfile for Python application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml for multi-service application
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
```

### 2. CI/CD Pipelines
Continuous Integration and Continuous Deployment automate testing and deployment processes.

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 .
        black --check .
        isort --check-only .
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t myapp:${{ github.sha }} .
        docker tag myapp:${{ github.sha }} myapp:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push myapp:${{ github.sha }}
        docker push myapp:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Deploy to cloud platform
        echo "Deploying to production..."
```

### 3. Cloud Deployment Strategies

#### AWS Deployment
```yaml
# AWS CloudFormation template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Python application on AWS'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']

  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for web application
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: myapp-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet
      SecurityGroups:
        - !Ref SecurityGroup

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: myapp-cluster

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: myapp-task
      Cpu: 256
      Memory: 512
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      ExecutionRoleArn: !Ref TaskExecutionRole
      ContainerDefinitions:
        - Name: myapp
          Image: myapp:latest
          PortMappings:
            - ContainerPort: 8000
              Protocol: tcp
          Environment:
            - Name: DATABASE_URL
              Value: !Ref DatabaseURL
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs

  Service:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: myapp-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref SecurityGroup
          Subnets:
            - !Ref PublicSubnet
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: myapp
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup
```

#### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-deployment
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: myapp-service
            port:
              number: 80
```

### 4. Infrastructure as Code

#### Terraform Configuration
```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.environment}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.environment}-igw"
    Environment = var.environment
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.environment}-public-subnet"
    Environment = var.environment
  }
}

# Security Group
resource "aws_security_group" "web" {
  name_prefix = "${var.environment}-web-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.environment}-web-sg"
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "${var.environment}-database"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "myapp"
  username = "myapp"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "production"
  
  tags = {
    Name        = "${var.environment}-database"
    Environment = var.environment
  }
}

# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}
```

#### Ansible Playbook
```yaml
# deploy.yml
---
- name: Deploy Python Application
  hosts: web_servers
  become: yes
  vars:
    app_name: myapp
    app_user: myapp
    app_dir: /opt/{{ app_name }}
    python_version: 3.9

  tasks:
    - name: Update system packages
      apt:
        update_cache: yes
        upgrade: yes

    - name: Install required packages
      apt:
        name:
          - python{{ python_version }}
          - python{{ python_version }}-pip
          - python{{ python_version }}-venv
          - nginx
          - supervisor
          - git
        state: present

    - name: Create application user
      user:
        name: "{{ app_user }}"
        system: yes
        shell: /bin/bash
        home: "{{ app_dir }}"
        create_home: yes

    - name: Create application directory
      file:
        path: "{{ app_dir }}"
        state: directory
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        mode: '0755'

    - name: Clone application repository
      git:
        repo: "{{ git_repository }}"
        dest: "{{ app_dir }}/src"
        version: "{{ git_branch }}"
      become_user: "{{ app_user }}"

    - name: Create virtual environment
      command: "python{{ python_version }} -m venv {{ app_dir }}/venv"
      args:
        creates: "{{ app_dir }}/venv"

    - name: Install Python dependencies
      pip:
        requirements: "{{ app_dir }}/src/requirements.txt"
        virtualenv: "{{ app_dir }}/venv"
        virtualenv_python: "python{{ python_version }}"

    - name: Create application configuration
      template:
        src: app.conf.j2
        dest: "{{ app_dir }}/app.conf"
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        mode: '0644'

    - name: Create systemd service
      template:
        src: myapp.service.j2
        dest: /etc/systemd/system/{{ app_name }}.service
        mode: '0644'
      notify: restart myapp

    - name: Configure Nginx
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
        mode: '0644'
      notify: restart nginx

    - name: Enable Nginx site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}
        dest: /etc/nginx/sites-enabled/{{ app_name }}
        state: link
      notify: restart nginx

    - name: Enable and start services
      systemd:
        name: "{{ item }}"
        enabled: yes
        state: started
      loop:
        - "{{ app_name }}"
        - nginx

  handlers:
    - name: restart myapp
      systemd:
        name: "{{ app_name }}"
        state: restarted

    - name: restart nginx
      systemd:
        name: nginx
        state: restarted
```

### 5. Monitoring and Logging

#### Prometheus and Grafana Setup
```yaml
# monitoring-stack.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
```

#### Application Monitoring
```python
# monitoring.py
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

def monitor_requests(func):
    """Decorator to monitor HTTP requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__, status='200').inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__, status='500').inc()
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

def setup_monitoring():
    """Setup monitoring and start metrics server"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Log startup
    logging.info("Monitoring started on port 8000")

# Health check endpoint
def health_check():
    """Health check endpoint for load balancer"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    }
```

### 6. Security Best Practices

#### Docker Security
```dockerfile
# Secure Dockerfile
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "app.py"]
```

#### Kubernetes Security
```yaml
# security-policy.yaml
apiVersion: v1
kind: PodSecurityPolicy
metadata:
  name: restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'

---
apiVersion: v1
kind: NetworkPolicy
metadata:
  name: myapp-network-policy
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
```

## Best Practices

### 1. Environment Management
```bash
# .env.example
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
LOG_LEVEL=INFO

# Environment-specific configurations
# development.env
DEBUG=True
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:password@localhost:5432/mydb_dev

# production.env
DEBUG=False
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:password@prod-db:5432/mydb_prod
```

### 2. Backup and Disaster Recovery
```python
# backup.py
import os
import boto3
from datetime import datetime
import subprocess

class BackupManager:
    def __init__(self, s3_bucket, aws_region='us-west-2'):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name=aws_region)
    
    def backup_database(self, db_config):
        """Backup PostgreSQL database"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"backup_{timestamp}.sql"
        
        # Create database dump
        dump_cmd = [
            'pg_dump',
            f"--host={db_config['host']}",
            f"--port={db_config['port']}",
            f"--username={db_config['user']}",
            f"--dbname={db_config['dbname']}",
            f"--file={backup_file}"
        ]
        
        subprocess.run(dump_cmd, check=True)
        
        # Upload to S3
        self.s3_client.upload_file(
            backup_file,
            self.s3_bucket,
            f"database-backups/{backup_file}"
        )
        
        # Clean up local file
        os.remove(backup_file)
        
        return f"s3://{self.s3_bucket}/database-backups/{backup_file}"
    
    def restore_database(self, s3_key, db_config):
        """Restore database from S3 backup"""
        backup_file = s3_key.split('/')[-1]
        
        # Download from S3
        self.s3_client.download_file(
            self.s3_bucket,
            s3_key,
            backup_file
        )
        
        # Restore database
        restore_cmd = [
            'psql',
            f"--host={db_config['host']}",
            f"--port={db_config['port']}",
            f"--username={db_config['user']}",
            f"--dbname={db_config['dbname']}",
            f"--file={backup_file}"
        ]
        
        subprocess.run(restore_cmd, check=True)
        
        # Clean up
        os.remove(backup_file)
```

### 3. Performance Optimization
```python
# performance.py
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
from functools import lru_cache

class PerformanceOptimizer:
    def __init__(self, redis_url='redis://localhost:6379'):
        self.redis_client = redis.from_url(redis_url)
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    @lru_cache(maxsize=128)
    def cached_expensive_operation(self, key):
        """Cache expensive operations"""
        # Simulate expensive operation
        time.sleep(1)
        return f"result_for_{key}"
    
    async def async_http_request(self, url):
        """Make async HTTP requests"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
    def cache_with_redis(self, key, value, ttl=3600):
        """Cache data in Redis"""
        self.redis_client.setex(key, ttl, value)
    
    def get_from_cache(self, key):
        """Get data from Redis cache"""
        return self.redis_client.get(key)
    
    def batch_process(self, items, batch_size=100):
        """Process items in batches"""
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            yield self.process_batch(batch)
    
    def process_batch(self, batch):
        """Process a batch of items"""
        # Process batch items
        return [item * 2 for item in batch]
```

## Quick Checks

### Check 1: Docker
```bash
# What will this command do?
docker run -d -p 8080:80 --name myapp nginx:alpine
```

### Check 2: Kubernetes
```yaml
# What will this create?
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: nginx
    image: nginx:alpine
    ports:
    - containerPort: 80
```

### Check 3: CI/CD
```yaml
# What triggers this workflow?
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

## Lab Problems

### Lab 1: Containerized Microservices
Build and deploy a microservices architecture using Docker and Kubernetes.

### Lab 2: CI/CD Pipeline
Create a complete CI/CD pipeline with automated testing, security scanning, and deployment.

### Lab 3: Cloud Migration
Migrate an on-premises application to a cloud platform with proper monitoring and scaling.

### Lab 4: Disaster Recovery
Implement a comprehensive backup and disaster recovery solution for a production system.

## AI Code Comparison
When working with AI-generated DevOps code, evaluate:
- **Security implementation** - are security best practices followed?
- **Resource optimization** - are resources allocated efficiently?
- **Error handling** - are failures handled gracefully?
- **Monitoring coverage** - are all critical metrics monitored?
- **Scalability design** - can the system handle increased load?

## Next Steps
- Learn about advanced Kubernetes features and operators
- Master cloud-native development patterns
- Explore service mesh technologies (Istio, Linkerd)
- Study infrastructure automation and GitOps
- Understand advanced security and compliance requirements
