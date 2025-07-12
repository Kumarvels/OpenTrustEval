#!/usr/bin/env python3
"""
OpenTrustEval MCP Server Configuration
Security settings and environment variables
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import json

class SecurityConfig(BaseSettings):
    """Security configuration for the MCP server"""
    
    # JWT Configuration
    jwt_secret: str = Field(
        default="your-super-secret-jwt-key-change-this-in-production",
        env="JWT_SECRET",
        description="Secret key for JWT token generation"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM",
        description="JWT algorithm"
    )
    jwt_expiry_hours: int = Field(
        default=24,
        env="JWT_EXPIRY_HOURS",
        description="JWT token expiry time in hours"
    )
    
    # API Configuration
    api_key_header: str = Field(
        default="X-API-Key",
        env="API_KEY_HEADER",
        description="Header name for API key"
    )
    
    # Encryption Configuration
    encryption_key: str = Field(
        default="",
        env="ENCRYPTION_KEY",
        description="Encryption key for sensitive data"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=1000,
        env="RATE_LIMIT_REQUESTS",
        description="Maximum requests per minute per client"
    )
    rate_limit_window: int = Field(
        default=60,
        env="RATE_LIMIT_WINDOW",
        description="Rate limit window in seconds"
    )
    
    # File Upload Configuration
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        env="MAX_FILE_SIZE",
        description="Maximum file size in bytes"
    )
    upload_dir: str = Field(
        default="./uploads",
        env="UPLOAD_DIR",
        description="Directory for file uploads"
    )
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["*"],
        env="ALLOWED_ORIGINS",
        description="Allowed CORS origins"
    )
    
    # SSL Configuration
    ssl_cert_path: Optional[str] = Field(
        default=None,
        env="SSL_CERT_PATH",
        description="Path to SSL certificate"
    )
    ssl_key_path: Optional[str] = Field(
        default=None,
        env="SSL_KEY_PATH",
        description="Path to SSL private key"
    )
    
    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        env="SERVER_HOST",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        env="SERVER_PORT",
        description="Server port"
    )
    workers: int = Field(
        default=1,
        env="SERVER_WORKERS",
        description="Number of worker processes"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./mcp_server.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    log_file: str = Field(
        default="mcp_server.log",
        env="LOG_FILE",
        description="Log file path"
    )
    
    # Monitoring Configuration
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=8001,
        env="METRICS_PORT",
        description="Metrics endpoint port"
    )
    
    # OpenTrustEval Configuration
    opentrusteval_data_dir: str = Field(
        default="./data_engineering",
        env="OPENTRUSTEVAL_DATA_DIR",
        description="OpenTrustEval data directory"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class APIConfig(BaseSettings):
    """API configuration for different endpoints"""
    
    # Trust Scoring API
    trust_score_timeout: int = Field(
        default=300,  # 5 minutes
        env="TRUST_SCORE_TIMEOUT",
        description="Timeout for trust scoring operations"
    )
    trust_score_max_retries: int = Field(
        default=3,
        env="TRUST_SCORE_MAX_RETRIES",
        description="Maximum retries for trust scoring"
    )
    
    # File Upload API
    upload_chunk_size: int = Field(
        default=1024 * 1024,  # 1MB
        env="UPLOAD_CHUNK_SIZE",
        description="Chunk size for file uploads"
    )
    upload_temp_dir: str = Field(
        default="./temp",
        env="UPLOAD_TEMP_DIR",
        description="Temporary directory for uploads"
    )
    
    # Cleanlab API
    cleanlab_timeout: int = Field(
        default=600,  # 10 minutes
        env="CLEANLAB_TIMEOUT",
        description="Timeout for Cleanlab operations"
    )
    
    # WebSocket API
    websocket_max_connections: int = Field(
        default=1000,
        env="WEBSOCKET_MAX_CONNECTIONS",
        description="Maximum WebSocket connections"
    )
    websocket_heartbeat_interval: int = Field(
        default=30,
        env="WEBSOCKET_HEARTBEAT_INTERVAL",
        description="WebSocket heartbeat interval in seconds"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentConfig(BaseSettings):
    """Development-specific configuration"""
    
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    reload: bool = Field(
        default=False,
        env="RELOAD",
        description="Enable auto-reload"
    )
    test_mode: bool = Field(
        default=False,
        env="TEST_MODE",
        description="Enable test mode"
    )
    
    # Test credentials
    test_username: str = Field(
        default="admin",
        env="TEST_USERNAME",
        description="Test username"
    )
    test_password: str = Field(
        default="admin123",
        env="TEST_PASSWORD",
        description="Test password"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global configuration instances
security_config = SecurityConfig()
api_config = APIConfig()
dev_config = DevelopmentConfig()

def get_config() -> dict:
    """Get complete configuration as dictionary"""
    return {
        "security": security_config.dict(),
        "api": api_config.dict(),
        "development": dev_config.dict()
    }

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check required directories
        upload_dir = Path(security_config.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        temp_dir = Path(api_config.upload_temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        # Check SSL files if SSL is enabled
        if security_config.ssl_cert_path and security_config.ssl_key_path:
            if not Path(security_config.ssl_cert_path).exists():
                raise ValueError(f"SSL certificate not found: {security_config.ssl_cert_path}")
            if not Path(security_config.ssl_key_path).exists():
                raise ValueError(f"SSL private key not found: {security_config.ssl_key_path}")
        
        # Check OpenTrustEval data directory
        opentrusteval_dir = Path(security_config.opentrusteval_data_dir)
        if not opentrusteval_dir.exists():
            raise ValueError(f"OpenTrustEval data directory not found: {opentrusteval_dir}")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def generate_env_template() -> str:
    """Generate environment file template"""
    template = """# OpenTrustEval MCP Server Environment Configuration

# Security Configuration
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
ENCRYPTION_KEY=your-encryption-key-here

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# File Upload
MAX_FILE_SIZE=104857600
UPLOAD_DIR=./uploads

# CORS
ALLOWED_ORIGINS=["*"]

# SSL (optional)
SSL_CERT_PATH=
SSL_KEY_PATH=

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1

# Database
DATABASE_URL=sqlite:///./mcp_server.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=mcp_server.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001

# OpenTrustEval
OPENTRUSTEVAL_DATA_DIR=../data_engineering

# API Configuration
TRUST_SCORE_TIMEOUT=300
TRUST_SCORE_MAX_RETRIES=3
UPLOAD_CHUNK_SIZE=1048576
UPLOAD_TEMP_DIR=./temp
CLEANLAB_TIMEOUT=600
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_HEARTBEAT_INTERVAL=30

# Development
DEBUG=false
RELOAD=false
TEST_MODE=false
TEST_USERNAME=admin
TEST_PASSWORD=admin123

# API Keys (comma-separated)
API_KEY_0=your-api-key-1
API_KEY_1=your-api-key-2
"""
    return template

if __name__ == "__main__":
    # Generate environment template
    env_template = generate_env_template()
    with open(".env.template", "w") as f:
        f.write(env_template)
    print("Environment template generated: .env.template")
    
    # Validate configuration
    if validate_config():
        print("Configuration validation passed")
        print("Configuration:")
        print(json.dumps(get_config(), indent=2))
    else:
        print("Configuration validation failed")
        exit(1) 