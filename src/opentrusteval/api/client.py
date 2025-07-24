#!/usr/bin/env python3
"""
OpenTrustEval MCP Client
Python client library for secure, high-speed communication with the MCP server
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol
import pandas as pd
import jwt
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPOpenTrustEvalClient:
    """Client for OpenTrustEval MCP Server"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: str = None, 
                 jwt_token: str = None, encryption_key: str = None):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key.encode()) if encryption_key else None
        self.session = None
        self.websocket = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'OpenTrustEval-MCP-Client/1.0.0'
        }
        
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        
        if self.jwt_token:
            headers['Authorization'] = f'Bearer {self.jwt_token}'
        
        return headers
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.fernet:
            return self.fernet.encrypt(data.encode()).decode()
        return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.fernet:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        return encrypted_data
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async with self.session.get(f"{self.server_url}/health") as response:
            return await response.json()
    
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and get JWT token"""
        data = {
            'username': username,
            'password': password
        }
        
        async with self.session.post(f"{self.server_url}/auth/login", json=data) as response:
            result = await response.json()
            if response.status == 200:
                self.jwt_token = result['token']
            return result
    
    async def calculate_trust_score(self, dataset_path: str, method: str = "ensemble", 
                                  features: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate trust score for a dataset"""
        data = {
            'dataset_path': dataset_path,
            'method': method,
            'features': features,
            'metadata': metadata
        }
        
        headers = self._get_headers()
        async with self.session.post(f"{self.server_url}/api/v1/trust-score", 
                                   json=data, headers=headers) as response:
            return await response.json()
    
    async def upload_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a file to the server"""
        # Read and encode file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encoded_data = base64.b64encode(file_data).decode()
        file_name = Path(file_path).name
        
        data = {
            'file_data': encoded_data,
            'file_name': file_name,
            'file_type': Path(file_path).suffix.lower(),
            'metadata': metadata or {}
        }
        
        headers = self._get_headers()
        async with self.session.post(f"{self.server_url}/api/v1/upload", 
                                   json=data, headers=headers) as response:
            return await response.json()
    
    async def cleanlab_comparison(self, dataset_path: str, cleanlab_option: int = 1,
                                comparison_method: str = "Side-by-Side", 
                                data_format: str = "CSV") -> Dict[str, Any]:
        """Run Cleanlab comparison"""
        data = {
            'dataset_path': dataset_path,
            'cleanlab_option': cleanlab_option,
            'comparison_method': comparison_method,
            'data_format': data_format
        }
        
        headers = self._get_headers()
        async with self.session.post(f"{self.server_url}/api/v1/cleanlab-compare", 
                                   json=data, headers=headers) as response:
            return await response.json()
    
    async def batch_process(self, dataset_paths: List[str], operation: str) -> Dict[str, Any]:
        """Process multiple datasets in batch"""
        data = {
            'dataset_paths': dataset_paths,
            'operation': operation
        }
        
        headers = self._get_headers()
        async with self.session.post(f"{self.server_url}/api/v1/batch-process", 
                                   json=data, headers=headers) as response:
            return await response.json()
    
    async def connect_websocket(self, client_id: str = None):
        """Connect to WebSocket for real-time communication"""
        if not client_id:
            client_id = str(uuid.uuid4())
        
        ws_url = self.server_url.replace('http', 'ws') + f"/ws/{client_id}"
        self.websocket = await websockets.connect(ws_url)
        logger.info(f"Connected to WebSocket with client ID: {client_id}")
        return client_id
    
    async def send_websocket_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send message via WebSocket and wait for response"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected. Call connect_websocket() first.")
        
        message = {
            "type": message_type,
            "request_id": str(uuid.uuid4()),
            **data
        }
        
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def websocket_trust_score(self, dataset_path: str, method: str = "ensemble") -> Dict[str, Any]:
        """Calculate trust score via WebSocket"""
        return await self.send_websocket_message("trust_score_request", {
            "dataset_path": dataset_path,
            "method": method
        })
    
    async def websocket_file_upload(self, file_path: str) -> Dict[str, Any]:
        """Upload file via WebSocket"""
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encoded_data = base64.b64encode(file_data).decode()
        file_name = Path(file_path).name
        
        return await self.send_websocket_message("file_upload_request", {
            "file_data": encoded_data,
            "file_name": file_name
        })
    
    async def websocket_cleanlab_comparison(self, dataset_path: str, cleanlab_option: int = 1,
                                          comparison_method: str = "Side-by-Side") -> Dict[str, Any]:
        """Run Cleanlab comparison via WebSocket"""
        return await self.send_websocket_message("cleanlab_comparison_request", {
            "dataset_path": dataset_path,
            "cleanlab_option": cleanlab_option,
            "comparison_method": comparison_method
        })

class SyncMCPOpenTrustEvalClient:
    """Synchronous wrapper for the MCP client"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: str = None, 
                 jwt_token: str = None, encryption_key: str = None):
        self.server_url = server_url
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.encryption_key = encryption_key
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    async def _get_client(self):
        """Get async client"""
        return MCPOpenTrustEvalClient(
            self.server_url, self.api_key, self.jwt_token, self.encryption_key
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async def _health_check():
            async with await self._get_client() as client:
                return await client.health_check()
        return self._run_async(_health_check())
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and get JWT token"""
        async def _login():
            async with await self._get_client() as client:
                result = await client.login(username, password)
                # Store the JWT token in the sync client
                if 'token' in result:
                    self.jwt_token = result['token']
                return result
        return self._run_async(_login())
    
    def calculate_trust_score(self, dataset_path: str, method: str = "ensemble", 
                            features: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate trust score for a dataset"""
        async def _trust_score():
            async with await self._get_client() as client:
                return await client.calculate_trust_score(dataset_path, method, features, metadata)
        return self._run_async(_trust_score())
    
    def upload_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a file to the server"""
        async def _upload():
            async with await self._get_client() as client:
                return await client.upload_file(file_path, metadata)
        return self._run_async(_upload())
    
    def cleanlab_comparison(self, dataset_path: str, cleanlab_option: int = 1,
                           comparison_method: str = "Side-by-Side", 
                           data_format: str = "CSV") -> Dict[str, Any]:
        """Run Cleanlab comparison"""
        async def _cleanlab():
            async with await self._get_client() as client:
                return await client.cleanlab_comparison(dataset_path, cleanlab_option, 
                                                      comparison_method, data_format)
        return self._run_async(_cleanlab())
    
    def batch_process(self, dataset_paths: List[str], operation: str) -> Dict[str, Any]:
        """Process multiple datasets in batch"""
        async def _batch():
            async with await self._get_client() as client:
                return await client.batch_process(dataset_paths, operation)
        return self._run_async(_batch())

# Example usage and testing
async def example_usage():
    """Example usage of the MCP client"""
    
    # Create client
    async with MCPOpenTrustEvalClient("http://localhost:8000") as client:
        
        # Check server health
        health = await client.health_check()
        print(f"Server health: {health}")
        
        # Login
        login_result = await client.login("admin", "admin123")
        print(f"Login result: {login_result}")
        
        # Calculate trust score
        trust_result = await client.calculate_trust_score(
            "path/to/dataset.csv", 
            method="ensemble"
        )
        print(f"Trust score: {trust_result}")
        
        # Upload file
        upload_result = await client.upload_file("path/to/file.csv")
        print(f"Upload result: {upload_result}")
        
        # Cleanlab comparison
        cleanlab_result = await client.cleanlab_comparison(
            "path/to/dataset.csv",
            cleanlab_option=1,
            comparison_method="Side-by-Side"
        )
        print(f"Cleanlab comparison: {cleanlab_result}")
        
        # WebSocket communication
        client_id = await client.connect_websocket()
        
        ws_trust_result = await client.websocket_trust_score("path/to/dataset.csv")
        print(f"WebSocket trust score: {ws_trust_result}")
        
        ws_upload_result = await client.websocket_file_upload("path/to/file.csv")
        print(f"WebSocket upload: {ws_upload_result}")

def sync_example_usage():
    """Synchronous example usage"""
    
    # Create sync client
    client = SyncMCPOpenTrustEvalClient("http://localhost:8000")
    
    # Check server health
    health = client.health_check()
    print(f"Server health: {health}")
    
    # Login
    login_result = client.login("admin", "admin123")
    print(f"Login result: {login_result}")
    
    # Calculate trust score
    trust_result = client.calculate_trust_score("path/to/dataset.csv")
    print(f"Trust score: {trust_result}")
    
    # Upload file
    upload_result = client.upload_file("path/to/file.csv")
    print(f"Upload result: {upload_result}")
    
    # Batch processing
    batch_result = client.batch_process(
        ["dataset1.csv", "dataset2.csv", "dataset3.csv"],
        "trust_score"
    )
    print(f"Batch processing: {batch_result}")

if __name__ == "__main__":
    # Run async example
    asyncio.run(example_usage())
    
    # Run sync example
    sync_example_usage() 