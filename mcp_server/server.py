#!/usr/bin/env python3
"""
OpenTrustEval MCP Server
Secure, high-speed bidirectional communication server for trust scoring system
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import ssl
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import websockets
from websockets.server import WebSocketServerProtocol
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import OpenTrustEval components
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard
from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
from data_engineering.cleanlab_integration import FallbackDataQualityManager
from data_engineering.dataset_integration import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    'JWT_SECRET': os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-this'),
    'JWT_ALGORITHM': 'HS256',
    'JWT_EXPIRY_HOURS': 24,
    'API_KEY_HEADER': 'X-API-Key',
    'ENCRYPTION_KEY': os.getenv('ENCRYPTION_KEY', Fernet.generate_key()),
    'RATE_LIMIT_REQUESTS': 1000,  # requests per minute
    'RATE_LIMIT_WINDOW': 60,  # seconds
    'MAX_FILE_SIZE': 100 * 1024 * 1024,  # 100MB
    'ALLOWED_ORIGINS': ['*'],  # Configure for production
    'SSL_CERT_PATH': os.getenv('SSL_CERT_PATH'),
    'SSL_KEY_PATH': os.getenv('SSL_KEY_PATH'),
}

class SecurityManager:
    """Manages security operations for the MCP server"""
    
    def __init__(self):
        self.fernet = Fernet(SECURITY_CONFIG['ENCRYPTION_KEY'])
        self.rate_limit_store = {}
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or file"""
        api_keys = {}
        # Load from environment variables
        for i in range(10):  # Support up to 10 API keys
            key_name = f'API_KEY_{i}'
            key_value = os.getenv(key_name)
            if key_value:
                api_keys[key_name] = key_value
        
        # Load from file if exists
        api_keys_file = Path('api_keys.json')
        if api_keys_file.exists():
            try:
                with open(api_keys_file, 'r') as f:
                    file_keys = json.load(f)
                    api_keys.update(file_keys)
            except Exception as e:
                logger.warning(f"Failed to load API keys from file: {e}")
        
        return api_keys
    
    def generate_jwt_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow().timestamp() + (SECURITY_CONFIG['JWT_EXPIRY_HOURS'] * 3600),
            'iat': datetime.utcnow().timestamp()
        }
        return jwt.encode(payload, SECURITY_CONFIG['JWT_SECRET'], algorithm=SECURITY_CONFIG['JWT_ALGORITHM'])
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, SECURITY_CONFIG['JWT_SECRET'], algorithms=[SECURITY_CONFIG['JWT_ALGORITHM']])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        return api_key in self.api_keys.values()
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_data(self, data: str) -> str:
        """Hash data for integrity checking"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit for client"""
        current_time = time.time()
        window_start = current_time - SECURITY_CONFIG['RATE_LIMIT_WINDOW']
        
        # Clean old entries
        self.rate_limit_store = {
            k: v for k, v in self.rate_limit_store.items() 
            if v['timestamp'] > window_start
        }
        
        # Check current client
        if client_id in self.rate_limit_store:
            client_data = self.rate_limit_store[client_id]
            if client_data['count'] >= SECURITY_CONFIG['RATE_LIMIT_REQUESTS']:
                return False
            client_data['count'] += 1
        else:
            self.rate_limit_store[client_id] = {
                'count': 1,
                'timestamp': current_time
            }
        
        return True

# Pydantic models for request/response
class TrustScoreRequest(BaseModel):
    """Request model for trust scoring"""
    dataset_path: str = Field(..., description="Path to dataset file")
    method: str = Field(default="ensemble", description="Scoring method")
    features: Optional[List[str]] = Field(None, description="Specific features to use")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FileUploadRequest(BaseModel):
    """Request model for file upload"""
    file_data: str = Field(..., description="Base64 encoded file data")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type/format")
    metadata: Optional[Dict[str, Any]] = Field(None, description="File metadata")

class CleanlabComparisonRequest(BaseModel):
    """Request model for Cleanlab comparison"""
    dataset_path: str = Field(..., description="Path to dataset")
    cleanlab_option: int = Field(..., ge=1, le=4, description="Cleanlab option (1-4)")
    comparison_method: str = Field(..., description="Comparison method")
    data_format: str = Field(default="CSV", description="Data format")

class TrustScoreResponse(BaseModel):
    """Response model for trust scoring"""
    request_id: str = Field(..., description="Unique request ID")
    trust_score: float = Field(..., description="Calculated trust score")
    method: str = Field(..., description="Method used")
    component_scores: Dict[str, float] = Field(..., description="Component scores")
    quality_metrics: Dict[str, float] = Field(..., description="Quality metrics")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Processing timestamp")

class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    request_id: str = Field(..., description="Unique request ID")
    file_path: str = Field(..., description="Saved file path")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="File hash for integrity")
    upload_time: float = Field(..., description="Upload time in seconds")
    status: str = Field(..., description="Upload status")

class CleanlabComparisonResponse(BaseModel):
    """Response model for Cleanlab comparison"""
    request_id: str = Field(..., description="Unique request ID")
    our_score: float = Field(..., description="Our trust score")
    cleanlab_score: float = Field(..., description="Cleanlab score")
    comparison_analysis: Dict[str, Any] = Field(..., description="Comparison analysis")
    statistical_test: Optional[Dict[str, Any]] = Field(None, description="Statistical test results")
    processing_time: float = Field(..., description="Processing time in seconds")

class LoginRequest(BaseModel):
    """Request model for login"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class BatchProcessRequest(BaseModel):
    """Request model for batch processing"""
    dataset_paths: List[str] = Field(..., description="List of dataset paths")
    operation: str = Field(..., description="Operation to perform")

class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request ID")
    timestamp: str = Field(..., description="Error timestamp")

class MCPOpenTrustEvalServer:
    """Main MCP server class for OpenTrustEval"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, ssl_enabled: bool = False):
        self.host = host
        self.port = port
        self.ssl_enabled = ssl_enabled
        self.security_manager = SecurityManager()
        self.dashboard = TrustScoringDashboard()
        self.advanced_engine = AdvancedTrustScoringEngine()
        self.fallback_manager = FallbackDataQualityManager()
        self.dataset_manager = DatasetManager()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="OpenTrustEval MCP Server",
            description="Secure, high-speed bidirectional communication server for trust scoring system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=SECURITY_CONFIG['ALLOWED_ORIGINS'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # WebSocket connections
        self.websocket_connections = {}
        
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health", tags=["Health"])
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "services": {
                    "trust_scoring": "active",
                    "file_upload": "active",
                    "cleanlab": "active",
                    "database": "active"
                }
            }
        
        # Authentication
        @self.app.post("/auth/login", tags=["Authentication"])
        async def login(request: LoginRequest):
            # In production, verify against user database
            if request.username == "admin" and request.password == "admin123":
                token = self.security_manager.generate_jwt_token(
                    request.username, ["trust_scoring", "file_upload", "cleanlab"]
                )
                return {"token": token, "user_id": request.username}
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Trust scoring endpoint
        @self.app.post("/api/v1/trust-score", response_model=TrustScoreResponse, tags=["Trust Scoring"])
        async def calculate_trust_score(
            request: TrustScoreRequest,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            try:
                # Verify authentication
                payload = self.security_manager.verify_jwt_token(credentials.credentials)
                
                # Check rate limit
                if not self.security_manager.check_rate_limit(payload['user_id']):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                request_id = str(uuid.uuid4())
                start_time = time.time()
                
                # Validate file exists
                if not os.path.exists(request.dataset_path):
                    raise HTTPException(status_code=404, detail="Dataset file not found")
                
                # Load data
                df = pd.read_csv(request.dataset_path)
                
                # Calculate trust score
                result = self.advanced_engine.calculate_advanced_trust_score(df)
                
                # Calculate quality metrics
                quality_result = self.fallback_manager.calculate_data_trust_score(df)
                
                # Ensure all component scores are valid floats
                component_scores = result.get('component_scores', {})
                cleaned_component_scores = {}
                for key, value in component_scores.items():
                    if value is not None:
                        try:
                            cleaned_component_scores[key] = float(value)
                        except (ValueError, TypeError):
                            cleaned_component_scores[key] = 0.0
                    else:
                        cleaned_component_scores[key] = 0.0
                
                processing_time = time.time() - start_time
                
                return TrustScoreResponse(
                    request_id=request_id,
                    trust_score=result.get('trust_score', 0.0),
                    method=result.get('method', 'ensemble'),
                    component_scores=cleaned_component_scores,
                    quality_metrics=quality_result.get('quality_metrics', {}),
                    processing_time=processing_time,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Trust scoring error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # File upload endpoint
        @self.app.post("/api/v1/upload", response_model=FileUploadResponse, tags=["File Upload"])
        async def upload_file(
            request: FileUploadRequest,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            try:
                # Verify authentication
                payload = self.security_manager.verify_jwt_token(credentials.credentials)
                
                # Check rate limit
                if not self.security_manager.check_rate_limit(payload['user_id']):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                request_id = str(uuid.uuid4())
                start_time = time.time()
                
                # Decode file data
                file_data = base64.b64decode(request.file_data)
                
                # Check file size
                if len(file_data) > SECURITY_CONFIG['MAX_FILE_SIZE']:
                    raise HTTPException(status_code=413, detail="File too large")
                
                # Create uploads directory
                uploads_dir = Path("./uploads")
                uploads_dir.mkdir(exist_ok=True)
                
                # Save file
                file_path = uploads_dir / request.file_name
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                
                # Calculate file hash
                file_hash = self.security_manager.hash_data(file_data.decode('latin-1'))
                
                upload_time = time.time() - start_time
                
                return FileUploadResponse(
                    request_id=request_id,
                    file_path=str(file_path),
                    file_size=len(file_data),
                    file_hash=file_hash,
                    upload_time=upload_time,
                    status="success"
                )
                
            except Exception as e:
                logger.error(f"File upload error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Cleanlab comparison endpoint
        @self.app.post("/api/v1/cleanlab-compare", response_model=CleanlabComparisonResponse, tags=["Cleanlab"])
        async def cleanlab_comparison(
            request: CleanlabComparisonRequest,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            try:
                # Verify authentication
                payload = self.security_manager.verify_jwt_token(credentials.credentials)
                
                # Check rate limit
                if not self.security_manager.check_rate_limit(payload['user_id']):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                request_id = str(uuid.uuid4())
                start_time = time.time()
                
                # Run comparison
                result = self.dashboard.run_cleanlab_comparison(
                    request.dataset_path,
                    request.data_format,
                    request.cleanlab_option,
                    request.comparison_method,
                    "Both"
                )
                
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                
                processing_time = time.time() - start_time
                
                return CleanlabComparisonResponse(
                    request_id=request_id,
                    our_score=result["our_score"],
                    cleanlab_score=result["cleanlab_score"],
                    comparison_analysis=result["comparison_analysis"],
                    statistical_test=result.get("comparison_analysis", {}).get("statistical_test"),
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Cleanlab comparison error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch processing endpoint
        @self.app.post("/api/v1/batch-process", tags=["Batch Processing"])
        async def batch_process(
            request: BatchProcessRequest,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            try:
                # Verify authentication
                payload = self.security_manager.verify_jwt_token(credentials.credentials)
                
                # Check rate limit
                if not self.security_manager.check_rate_limit(payload['user_id']):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                request_id = str(uuid.uuid4())
                results = []
                
                for dataset_path in request.dataset_paths:
                    try:
                        if request.operation == "trust_score":
                            df = pd.read_csv(dataset_path)
                            result = self.advanced_engine.calculate_advanced_trust_score(df)
                            results.append({
                                "dataset": dataset_path,
                                "trust_score": result.get('trust_score', 0.0),
                                "status": "success"
                            })
                        elif request.operation == "quality_assessment":
                            df = pd.read_csv(dataset_path)
                            result = self.fallback_manager.calculate_data_trust_score(df)
                            results.append({
                                "dataset": dataset_path,
                                "quality_metrics": result.get('quality_metrics', {}),
                                "status": "success"
                            })
                        else:
                            results.append({
                                "dataset": dataset_path,
                                "status": "error",
                                "error": f"Unknown operation: {request.operation}"
                            })
                    except Exception as e:
                        results.append({
                            "dataset": dataset_path,
                            "status": "error",
                            "error": str(e)
                        })
                
                return {
                    "request_id": request_id,
                    "operation": request.operation,
                    "total_datasets": len(request.dataset_paths),
                    "successful": len([r for r in results if r["status"] == "success"]),
                    "failed": len([r for r in results if r["status"] == "error"]),
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint for real-time communication
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(self, websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.websocket_connections[client_id] = websocket
            
            try:
                while True:
                    # Receive message
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Process message based on type
                    if data.get("type") == "trust_score_request":
                        response = await self._handle_websocket_trust_score(data, client_id)
                    elif data.get("type") == "file_upload_request":
                        response = await self._handle_websocket_file_upload(data, client_id)
                    elif data.get("type") == "cleanlab_comparison_request":
                        response = await self._handle_websocket_cleanlab(data, client_id)
                    else:
                        response = {
                            "type": "error",
                            "error": "Unknown message type",
                            "request_id": data.get("request_id")
                        }
                    
                    # Send response
                    await websocket.send(json.dumps(response))
                    
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                if client_id in self.websocket_connections:
                    del self.websocket_connections[client_id]
    
    async def _handle_websocket_trust_score(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle trust score request via WebSocket"""
        try:
            request_id = data.get("request_id", str(uuid.uuid4()))
            dataset_path = data.get("dataset_path")
            method = data.get("method", "ensemble")
            
            if not dataset_path or not os.path.exists(dataset_path):
                return {
                    "type": "trust_score_response",
                    "request_id": request_id,
                    "error": "Dataset file not found"
                }
            
            df = pd.read_csv(dataset_path)
            result = self.advanced_engine.calculate_advanced_trust_score(df)
            
            return {
                "type": "trust_score_response",
                "request_id": request_id,
                "trust_score": result.get('trust_score', 0.0),
                "method": result.get('method', 'ensemble'),
                "component_scores": result.get('component_scores', {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "type": "trust_score_response",
                "request_id": data.get("request_id"),
                "error": str(e)
            }
    
    async def _handle_websocket_file_upload(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle file upload request via WebSocket"""
        try:
            request_id = data.get("request_id", str(uuid.uuid4()))
            file_data = data.get("file_data")
            file_name = data.get("file_name")
            
            if not file_data or not file_name:
                return {
                    "type": "file_upload_response",
                    "request_id": request_id,
                    "error": "Missing file data or name"
                }
            
            # Decode and save file
            file_bytes = base64.b64decode(file_data)
            uploads_dir = Path("./uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            file_path = uploads_dir / file_name
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            
            return {
                "type": "file_upload_response",
                "request_id": request_id,
                "file_path": str(file_path),
                "file_size": len(file_bytes),
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "type": "file_upload_response",
                "request_id": data.get("request_id"),
                "error": str(e)
            }
    
    async def _handle_websocket_cleanlab(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle Cleanlab comparison request via WebSocket"""
        try:
            request_id = data.get("request_id", str(uuid.uuid4()))
            dataset_path = data.get("dataset_path")
            cleanlab_option = data.get("cleanlab_option", 1)
            comparison_method = data.get("comparison_method", "Side-by-Side")
            
            if not dataset_path or not os.path.exists(dataset_path):
                return {
                    "type": "cleanlab_comparison_response",
                    "request_id": request_id,
                    "error": "Dataset file not found"
                }
            
            result = self.dashboard.run_cleanlab_comparison(
                dataset_path, "CSV", cleanlab_option, comparison_method, "Both"
            )
            
            if "error" in result:
                return {
                    "type": "cleanlab_comparison_response",
                    "request_id": request_id,
                    "error": result["error"]
                }
            
            return {
                "type": "cleanlab_comparison_response",
                "request_id": request_id,
                "our_score": result["our_score"],
                "cleanlab_score": result["cleanlab_score"],
                "comparison_analysis": result["comparison_analysis"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "type": "cleanlab_comparison_response",
                "request_id": data.get("request_id"),
                "error": str(e)
            }
    
    def run(self, host: str = None, port: int = None, ssl_enabled: bool = None):
        """Run the MCP server"""
        host = host or self.host
        port = port or self.port
        ssl_enabled = ssl_enabled if ssl_enabled is not None else self.ssl_enabled
        
        logger.info(f"Starting OpenTrustEval MCP Server on {host}:{port}")
        logger.info(f"SSL enabled: {ssl_enabled}")
        logger.info(f"Documentation available at: http://{host}:{port}/docs")
        
        ssl_context = None
        if ssl_enabled and SECURITY_CONFIG['SSL_CERT_PATH'] and SECURITY_CONFIG['SSL_KEY_PATH']:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                SECURITY_CONFIG['SSL_CERT_PATH'],
                SECURITY_CONFIG['SSL_KEY_PATH']
            )
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_keyfile=SECURITY_CONFIG['SSL_KEY_PATH'] if ssl_enabled else None,
            ssl_certfile=SECURITY_CONFIG['SSL_CERT_PATH'] if ssl_enabled else None,
            log_level="info"
        )

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenTrustEval MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--ssl", action="store_true", help="Enable SSL")
    parser.add_argument("--cert", help="SSL certificate path")
    parser.add_argument("--key", help="SSL private key path")
    
    args = parser.parse_args()
    
    # Update SSL config if provided
    if args.cert:
        SECURITY_CONFIG['SSL_CERT_PATH'] = args.cert
    if args.key:
        SECURITY_CONFIG['SSL_KEY_PATH'] = args.key
    
    # Create and run server
    server = MCPOpenTrustEvalServer(args.host, args.port, args.ssl)
    server.run()

if __name__ == "__main__":
    main() 