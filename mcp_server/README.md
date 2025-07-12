# OpenTrustEval MCP Server

## 🚀 Overview

The OpenTrustEval MCP (Model Context Protocol) Server provides secure, high-speed bidirectional communication for the OpenTrustEval trust scoring system. It enables external applications to interact with the trust scoring engine programmatically through REST APIs and WebSocket connections.

![MCP Server](https://img.shields.io/badge/MCP%20Server-FastAPI-blue)
![Security](https://img.shields.io/badge/Security-JWT%20%7C%20Encryption-green)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange)
![SSL](https://img.shields.io/badge/SSL-Supported-red)

## ✨ Features

### 🔐 **Security Features**
- **JWT Authentication**: Secure token-based authentication
- **API Key Support**: Multiple API key management
- **Data Encryption**: End-to-end encryption for sensitive data
- **Rate Limiting**: Configurable rate limiting per client
- **SSL/TLS Support**: Secure HTTPS communication
- **CORS Protection**: Configurable cross-origin resource sharing

### 🌐 **Communication Protocols**
- **REST API**: Standard HTTP REST endpoints
- **WebSocket**: Real-time bidirectional communication
- **Async Support**: Full async/await support
- **Batch Processing**: Efficient batch operations

### 📊 **Trust Scoring Integration**
- **Trust Score Calculation**: Advanced trust scoring methods
- **File Upload**: Multi-format file upload support
- **Cleanlab Comparison**: Benchmarking against industry standards
- **Quality Assessment**: Comprehensive data quality evaluation
- **Batch Processing**: Process multiple datasets efficiently

### 📈 **Monitoring & Analytics**
- **Health Checks**: Server health monitoring
- **Metrics**: Prometheus metrics integration
- **Logging**: Comprehensive logging system
- **Performance**: High-performance async processing

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenTrustEval system installed
- pip package manager

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/OpenTrustEval.git
cd OpenTrustEval

# Install MCP server dependencies
pip install -r mcp_server/requirements.txt

# Install OpenTrustEval dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
# Generate environment template
cd mcp_server
python config.py

# Copy and edit environment file
cp .env.template .env
# Edit .env with your configuration
```

### SSL Certificate Setup (Optional)
```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update .env file
SSL_CERT_PATH=cert.pem
SSL_KEY_PATH=key.pem
```

## 🚀 Quick Start

### 1. Start the MCP Server
```bash
# Basic startup
python mcp_server/server.py

# With custom configuration
python mcp_server/server.py --host 0.0.0.0 --port 8000 --ssl

# Using environment variables
SERVER_HOST=0.0.0.0 SERVER_PORT=8000 python mcp_server/server.py
```

### 2. Test the Server
```bash
# Check server health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

### 3. Use the Client Library
```python
from mcp_server.client import SyncMCPOpenTrustEvalClient

# Create client
client = SyncMCPOpenTrustEvalClient("http://localhost:8000")

# Login
login_result = client.login("admin", "admin123")

# Calculate trust score
result = client.calculate_trust_score("path/to/dataset.csv")
print(f"Trust Score: {result['trust_score']}")
```

## 📚 API Reference

### Authentication

#### Login
```http
POST /auth/login
Content-Type: application/json

{
    "username": "admin",
    "password": "admin123"
}
```

**Response:**
```json
{
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "user_id": "admin"
}
```

### Trust Scoring API

#### Calculate Trust Score
```http
POST /api/v1/trust-score
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "dataset_path": "path/to/dataset.csv",
    "method": "ensemble",
    "features": ["feature1", "feature2"],
    "metadata": {"source": "api", "version": "1.0"}
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "trust_score": 0.85,
    "method": "ensemble",
    "component_scores": {
        "data_quality": 0.9,
        "statistical_robustness": 0.8,
        "anomaly_detection": 0.85
    },
    "quality_metrics": {
        "missing_values_ratio": 0.02,
        "duplicate_rows_ratio": 0.01,
        "outlier_ratio": 0.05
    },
    "processing_time": 2.34,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### File Upload API

#### Upload File
```http
POST /api/v1/upload
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "file_data": "base64-encoded-file-content",
    "file_name": "dataset.csv",
    "file_type": ".csv",
    "metadata": {"description": "Customer data", "version": "1.0"}
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "file_path": "./uploads/dataset.csv",
    "file_size": 1024000,
    "file_hash": "sha256-hash",
    "upload_time": 1.23,
    "status": "success"
}
```

### Cleanlab Comparison API

#### Run Cleanlab Comparison
```http
POST /api/v1/cleanlab-compare
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "dataset_path": "path/to/dataset.csv",
    "cleanlab_option": 1,
    "comparison_method": "Side-by-Side",
    "data_format": "CSV"
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "our_score": 0.85,
    "cleanlab_score": 0.82,
    "comparison_analysis": {
        "score_difference": 0.03,
        "score_ratio": 1.037,
        "percentage_difference": 3.66
    },
    "statistical_test": {
        "t_statistic": 2.45,
        "p_value": 0.015,
        "significant": true
    },
    "processing_time": 5.67
}
```

### Batch Processing API

#### Batch Process
```http
POST /api/v1/batch-process
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "dataset_paths": [
        "dataset1.csv",
        "dataset2.csv",
        "dataset3.csv"
    ],
    "operation": "trust_score"
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "operation": "trust_score",
    "total_datasets": 3,
    "successful": 2,
    "failed": 1,
    "results": [
        {
            "dataset": "dataset1.csv",
            "trust_score": 0.85,
            "status": "success"
        },
        {
            "dataset": "dataset2.csv",
            "trust_score": 0.78,
            "status": "success"
        },
        {
            "dataset": "dataset3.csv",
            "status": "error",
            "error": "File not found"
        }
    ],
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🔌 WebSocket API

### Connection
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/client-id-123');
```

### Message Format
```json
{
    "type": "trust_score_request",
    "request_id": "uuid-string",
    "dataset_path": "path/to/dataset.csv",
    "method": "ensemble"
}
```

### Response Format
```json
{
    "type": "trust_score_response",
    "request_id": "uuid-string",
    "trust_score": 0.85,
    "method": "ensemble",
    "component_scores": {...},
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Message Types
- `trust_score_request`: Calculate trust score
- `file_upload_request`: Upload file
- `cleanlab_comparison_request`: Run Cleanlab comparison

## 🐍 Python Client Library

### Async Client
```python
import asyncio
from mcp_server.client import MCPOpenTrustEvalClient

async def main():
    async with MCPOpenTrustEvalClient("http://localhost:8000") as client:
        # Login
        await client.login("admin", "admin123")
        
        # Calculate trust score
        result = await client.calculate_trust_score("dataset.csv")
        print(f"Trust Score: {result['trust_score']}")
        
        # Upload file
        upload_result = await client.upload_file("data.csv")
        print(f"Uploaded: {upload_result['file_path']}")
        
        # WebSocket communication
        client_id = await client.connect_websocket()
        ws_result = await client.websocket_trust_score("dataset.csv")
        print(f"WebSocket result: {ws_result}")

asyncio.run(main())
```

### Sync Client
```python
from mcp_server.client import SyncMCPOpenTrustEvalClient

# Create client
client = SyncMCPOpenTrustEvalClient("http://localhost:8000")

# Login
client.login("admin", "admin123")

# Calculate trust score
result = client.calculate_trust_score("dataset.csv")
print(f"Trust Score: {result['trust_score']}")

# Batch processing
batch_result = client.batch_process(
    ["dataset1.csv", "dataset2.csv"],
    "trust_score"
)
print(f"Batch results: {batch_result}")
```

## 🔧 Configuration

### Environment Variables

#### Security Configuration
```bash
# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
ENCRYPTION_KEY=your-encryption-key

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# File Upload
MAX_FILE_SIZE=104857600
UPLOAD_DIR=./uploads
```

#### Server Configuration
```bash
# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1

# SSL Configuration
SSL_CERT_PATH=cert.pem
SSL_KEY_PATH=key.pem

# CORS
ALLOWED_ORIGINS=["*"]
```

#### API Configuration
```bash
# Timeouts
TRUST_SCORE_TIMEOUT=300
CLEANLAB_TIMEOUT=600

# WebSocket
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_HEARTBEAT_INTERVAL=30
```

### Configuration Validation
```bash
# Validate configuration
python mcp_server/config.py

# Generate environment template
python mcp_server/config.py
```

## 🔒 Security

### Authentication Methods

#### JWT Authentication
```python
# Login to get JWT token
response = await client.login("username", "password")
token = response["token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
```

#### API Key Authentication
```python
# Set API key
client = MCPOpenTrustEvalClient(
    "http://localhost:8000",
    api_key="your-api-key"
)
```

### Data Encryption
```python
# Enable encryption
client = MCPOpenTrustEvalClient(
    "http://localhost:8000",
    encryption_key="your-encryption-key"
)

# Data is automatically encrypted/decrypted
```

### Rate Limiting
- **Default**: 1000 requests per minute per client
- **Configurable**: Via environment variables
- **Per-client**: Based on user ID or IP address

### SSL/TLS
```bash
# Enable SSL
python mcp_server/server.py --ssl

# With custom certificates
SSL_CERT_PATH=cert.pem SSL_KEY_PATH=key.pem python mcp_server/server.py
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0",
    "services": {
        "trust_scoring": "active",
        "file_upload": "active",
        "cleanlab": "active",
        "database": "active"
    }
}
```

### Metrics (Prometheus)
```bash
# Enable metrics
ENABLE_METRICS=true METRICS_PORT=8001 python mcp_server/server.py

# Access metrics
curl http://localhost:8001/metrics
```

### Logging
```bash
# Configure logging level
LOG_LEVEL=DEBUG python mcp_server/server.py

# Log file
tail -f mcp_server.log
```

## 🧪 Testing

### Unit Tests
```bash
# Run tests
pytest mcp_server/tests/

# With coverage
pytest --cov=mcp_server mcp_server/tests/
```

### Integration Tests
```bash
# Test server endpoints
python mcp_server/test_integration.py

# Test client library
python mcp_server/test_client.py
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f mcp_server/locustfile.py --host=http://localhost:8000
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "mcp_server/server.py"]
```

```bash
# Build and run
docker build -t opentrusteval-mcp .
docker run -p 8000:8000 opentrusteval-mcp
```

### Production Deployment
```bash
# Using gunicorn
pip install gunicorn
gunicorn mcp_server.server:app -w 4 -k uvicorn.workers.UvicornWorker

# Using systemd
sudo systemctl enable opentrusteval-mcp
sudo systemctl start opentrusteval-mcp
```

### Load Balancer Configuration
```nginx
# Nginx configuration
upstream mcp_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name mcp.yourdomain.com;
    
    location / {
        proxy_pass http://mcp_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws/ {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🔧 Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if server is running
netstat -tlnp | grep 8000

# Check logs
tail -f mcp_server.log
```

#### Authentication Errors
```bash
# Check JWT token
python -c "import jwt; print(jwt.decode('your-token', 'your-secret', algorithms=['HS256']))"

# Verify API key
curl -H "X-API-Key: your-key" http://localhost:8000/health
```

#### File Upload Issues
```bash
# Check file size limits
ls -lh your-file.csv

# Check upload directory permissions
ls -la ./uploads/
```

#### Performance Issues
```bash
# Monitor server resources
htop
iotop

# Check rate limiting
curl -H "X-RateLimit-Remaining: 0" http://localhost:8000/api/v1/trust-score
```

### Debug Mode
```bash
# Enable debug mode
DEBUG=true python mcp_server/server.py

# Verbose logging
LOG_LEVEL=DEBUG python mcp_server/server.py
```

## 📞 Support

### Getting Help
- **Documentation**: Check this README first
- **API Docs**: Visit http://localhost:8000/docs
- **Issues**: Report on GitHub
- **Discussions**: Use GitHub Discussions

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎯 Quick Reference

### Server URLs
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8001/metrics (if enabled)

### Key Commands
```bash
# Start server
python mcp_server/server.py

# Start with SSL
python mcp_server/server.py --ssl

# Test client
python mcp_server/client.py

# Validate config
python mcp_server/config.py
```

### Environment Setup
```bash
# Generate template
python mcp_server/config.py

# Copy and edit
cp .env.template .env
# Edit .env file
```

### Client Usage
```python
# Quick start
from mcp_server.client import SyncMCPOpenTrustEvalClient
client = SyncMCPOpenTrustEvalClient("http://localhost:8000")
result = client.calculate_trust_score("dataset.csv")
``` 