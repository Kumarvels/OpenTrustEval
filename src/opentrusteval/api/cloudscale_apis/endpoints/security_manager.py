from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile, Form, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
import logging
from datetime import datetime, timedelta
import hashlib
import pickle
from functools import lru_cache
import time
import jwt
import bcrypt
import secrets
import re
from pathlib import Path

# Import security components
try:
    from src.opentrusteval.security.auth_manager import AuthManager
    from src.opentrusteval.security.security_monitor import SecurityMonitor
    from src.opentrusteval.security.dependency_scanner import DependencyScanner
    from src.opentrusteval.security.secrets_manager import SecretsManager
    from src.opentrusteval.security.oauth_provider import OAuthProvider
    from src.opentrusteval.security.saml_provider import SAMLProvider
    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Security components not available: {e}")
    SECURITY_AVAILABLE = False

router = APIRouter(prefix="/security", tags=["Security Management"])

# Global instances with caching
_auth_manager = None
_security_monitor = None
_dependency_scanner = None
_secrets_manager = None
_oauth_provider = None
_saml_provider = None
_cache = {}
_cache_timestamps = {}

# Security settings
SECURITY_CONFIG = {
    "jwt_secret": "your-super-secret-jwt-key-change-in-production",
    "jwt_algorithm": "HS256",
    "jwt_expiry_hours": 24,
    "bcrypt_rounds": 12,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 30,
    "password_min_length": 8,
    "session_timeout_minutes": 60
}

# Rate limiting
rate_limit_store = {}
rate_limit_config = {
    "max_requests_per_minute": 100,
    "max_requests_per_hour": 1000
}

def get_auth_manager():
    """Get or create AuthManager instance"""
    global _auth_manager
    if _auth_manager is None and SECURITY_AVAILABLE:
        _auth_manager = AuthManager()
    return _auth_manager

def get_security_monitor():
    """Get or create SecurityMonitor instance"""
    global _security_monitor
    if _security_monitor is None and SECURITY_AVAILABLE:
        _security_monitor = SecurityMonitor()
    return _security_monitor

def get_dependency_scanner():
    """Get or create DependencyScanner instance"""
    global _dependency_scanner
    if _dependency_scanner is None and SECURITY_AVAILABLE:
        _dependency_scanner = DependencyScanner()
    return _dependency_scanner

def get_secrets_manager():
    """Get or create SecretsManager instance"""
    global _secrets_manager
    if _secrets_manager is None and SECURITY_AVAILABLE:
        _secrets_manager = SecretsManager()
    return _secrets_manager

def get_oauth_provider():
    """Get or create OAuthProvider instance"""
    global _oauth_provider
    if _oauth_provider is None and SECURITY_AVAILABLE:
        _oauth_provider = OAuthProvider()
    return _oauth_provider

def get_saml_provider():
    """Get or create SAMLProvider instance"""
    global _saml_provider
    if _saml_provider is None and SECURITY_AVAILABLE:
        _saml_provider = SAMLProvider()
    return _saml_provider

# Cache management
def get_cache_key(operation: str, params: Dict) -> str:
    """Generate cache key for operation"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{operation}:{param_str}".encode()).hexdigest()

def is_cache_valid(cache_key: str, max_age: int = 300) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    return time.time() - _cache_timestamps[cache_key] < max_age

def set_cache(cache_key: str, data: Any):
    """Set cache entry with timestamp"""
    _cache[cache_key] = data
    _cache_timestamps[cache_key] = time.time()

def get_cache(cache_key: str):
    """Get cache entry if valid"""
    if is_cache_valid(cache_key):
        return _cache.get(cache_key)
    return None

# Rate limiting
def check_rate_limit(client_id: str, endpoint: str) -> bool:
    """Check if client has exceeded rate limits"""
    current_time = time.time()
    key = f"{client_id}:{endpoint}"
    
    if key not in rate_limit_store:
        rate_limit_store[key] = {"requests": [], "hourly": []}
    
    # Clean old requests
    minute_ago = current_time - 60
    hour_ago = current_time - 3600
    
    rate_limit_store[key]["requests"] = [
        req_time for req_time in rate_limit_store[key]["requests"] 
        if req_time > minute_ago
    ]
    rate_limit_store[key]["hourly"] = [
        req_time for req_time in rate_limit_store[key]["hourly"] 
        if req_time > hour_ago
    ]
    
    # Check limits
    if (len(rate_limit_store[key]["requests"]) >= rate_limit_config["max_requests_per_minute"] or
        len(rate_limit_store[key]["hourly"]) >= rate_limit_config["max_requests_per_hour"]):
        return False
    
    # Add current request
    rate_limit_store[key]["requests"].append(current_time)
    rate_limit_store[key]["hourly"].append(current_time)
    
    return True

# JWT token management
def create_jwt_token(user_id: str, username: str, roles: List[str]) -> str:
    """Create JWT token for user"""
    payload = {
        "user_id": user_id,
        "username": username,
        "roles": roles,
        "exp": datetime.utcnow() + timedelta(hours=SECURITY_CONFIG["jwt_expiry_hours"]),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECURITY_CONFIG["jwt_secret"], algorithm=SECURITY_CONFIG["jwt_algorithm"])

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECURITY_CONFIG["jwt_secret"], algorithms=[SECURITY_CONFIG["jwt_algorithm"]])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication dependency
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    return payload

# Health check
@router.get("/health")
async def security_health_check():
    """Health check for security management system"""
    try:
        auth_manager = get_auth_manager()
        security_monitor = get_security_monitor()
        
        return {
            "status": "healthy",
            "security_available": SECURITY_AVAILABLE,
            "auth_manager_ready": auth_manager is not None,
            "security_monitor_ready": security_monitor is not None,
            "cache_entries": len(_cache),
            "rate_limit_entries": len(rate_limit_store),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Authentication Operations
@router.post("/auth/register")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    full_name: Optional[str] = Form(None),
    roles: Optional[str] = Form("user")
):
    """Register a new user"""
    try:
        # Rate limiting
        client_id = f"register:{username}"
        if not check_rate_limit(client_id, "register"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Validate password strength
        if len(password) < SECURITY_CONFIG["password_min_length"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {SECURITY_CONFIG['password_min_length']} characters long"
            )
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Parse roles
        role_list = json.loads(roles) if isinstance(roles, str) else [roles]
        
        # Register user
        user_id = auth_manager.register_user(username, email, password, full_name, role_list)
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "email": email,
            "roles": role_list,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")

@router.post("/auth/login")
async def login_user(
    username: str = Form(...),
    password: str = Form(...)
):
    """Login user and return JWT token"""
    try:
        # Rate limiting
        client_id = f"login:{username}"
        if not check_rate_limit(client_id, "login"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Authenticate user
        user = auth_manager.authenticate_user(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create JWT token
        token = create_jwt_token(user["user_id"], user["username"], user["roles"])
        
        return {
            "success": True,
            "token": token,
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "roles": user["roles"],
                "full_name": user.get("full_name")
            },
            "expires_in": SECURITY_CONFIG["jwt_expiry_hours"] * 3600,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to login: {str(e)}")

@router.post("/auth/logout")
async def logout_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user (invalidate token)"""
    try:
        # In a real implementation, you would add the token to a blacklist
        # For now, we'll just return success
        return {
            "success": True,
            "message": "Successfully logged out",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to logout: {str(e)}")

@router.get("/auth/profile")
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user profile"""
    try:
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        user_profile = auth_manager.get_user_profile(current_user["user_id"])
        
        return {
            "success": True,
            "profile": user_profile,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")

@router.put("/auth/profile")
async def update_user_profile(
    full_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update current user profile"""
    try:
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Validate email if provided
        if email:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                raise HTTPException(status_code=400, detail="Invalid email format")
        
        updated_profile = auth_manager.update_user_profile(
            current_user["user_id"], 
            full_name=full_name, 
            email=email
        )
        
        return {
            "success": True,
            "profile": updated_profile,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

# User Management Operations
@router.get("/users/list")
async def list_users(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = Query(100, description="Maximum number of users to return"),
    offset: int = Query(0, description="Number of users to skip"),
    role_filter: Optional[str] = Query(None, description="Filter by role")
):
    """List all users (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Check cache
        cache_key = get_cache_key("list_users", {"limit": limit, "offset": offset, "role_filter": role_filter})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        users = auth_manager.list_users(limit=limit, offset=offset, role_filter=role_filter)
        
        result = {
            "success": True,
            "users": users,
            "total_count": len(users),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user details (admin or self)"""
    try:
        # Check permissions
        if user_id != current_user["user_id"] and "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Check cache
        cache_key = get_cache_key("get_user", {"user_id": user_id})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        user = auth_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        result = {
            "success": True,
            "user": user,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")

@router.put("/users/{user_id}/roles")
async def update_user_roles(
    user_id: str,
    roles: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user roles (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        # Parse roles
        role_list = json.loads(roles) if isinstance(roles, str) else [roles]
        
        updated_user = auth_manager.update_user_roles(user_id, role_list)
        
        return {
            "success": True,
            "user": updated_user,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user roles: {str(e)}")

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete user (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        auth_manager = get_auth_manager()
        if not auth_manager:
            raise HTTPException(status_code=503, detail="Auth manager not available")
        
        success = auth_manager.delete_user(user_id)
        
        return {
            "success": True,
            "deleted": success,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")

# Security Monitoring Operations
@router.get("/monitor/events")
async def get_security_events(
    current_user: Dict[str, Any] = Depends(get_current_user),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Maximum number of events to return"),
    offset: int = Query(0, description="Number of events to skip")
):
    """Get security events (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        security_monitor = get_security_monitor()
        if not security_monitor:
            raise HTTPException(status_code=503, detail="Security monitor not available")
        
        # Check cache
        cache_key = get_cache_key("security_events", {
            "event_type": event_type, "severity": severity, "limit": limit, "offset": offset
        })
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        events = security_monitor.get_events(
            event_type=event_type,
            severity=severity,
            limit=limit,
            offset=offset
        )
        
        result = {
            "success": True,
            "events": events,
            "total_count": len(events),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security events: {str(e)}")

@router.get("/monitor/alerts")
async def get_security_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by alert status"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
    offset: int = Query(0, description="Number of alerts to skip")
):
    """Get security alerts (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        security_monitor = get_security_monitor()
        if not security_monitor:
            raise HTTPException(status_code=503, detail="Security monitor not available")
        
        # Check cache
        cache_key = get_cache_key("security_alerts", {
            "status": status, "limit": limit, "offset": offset
        })
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        alerts = security_monitor.get_alerts(
            status=status,
            limit=limit,
            offset=offset
        )
        
        result = {
            "success": True,
            "alerts": alerts,
            "total_count": len(alerts),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security alerts: {str(e)}")

@router.post("/monitor/scan")
async def run_security_scan(
    scan_type: str = Form(..., description="Type of security scan"),
    target: Optional[str] = Form(None, description="Scan target"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Run security scan (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        security_monitor = get_security_monitor()
        if not security_monitor:
            raise HTTPException(status_code=503, detail="Security monitor not available")
        
        scan_result = security_monitor.run_scan(scan_type, target)
        
        return {
            "success": True,
            "scan_type": scan_type,
            "target": target,
            "result": scan_result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run security scan: {str(e)}")

# Dependency Scanning Operations
@router.post("/dependencies/scan")
async def scan_dependencies(
    project_path: Optional[str] = Form(None, description="Project path to scan"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Scan project dependencies for vulnerabilities"""
    try:
        dependency_scanner = get_dependency_scanner()
        if not dependency_scanner:
            raise HTTPException(status_code=503, detail="Dependency scanner not available")
        
        scan_result = dependency_scanner.scan_project(project_path or ".")
        
        return {
            "success": True,
            "project_path": project_path or ".",
            "vulnerabilities": scan_result.get("vulnerabilities", []),
            "total_vulnerabilities": len(scan_result.get("vulnerabilities", [])),
            "critical_count": len([v for v in scan_result.get("vulnerabilities", []) if v.get("severity") == "critical"]),
            "high_count": len([v for v in scan_result.get("vulnerabilities", []) if v.get("severity") == "high"]),
            "medium_count": len([v for v in scan_result.get("vulnerabilities", []) if v.get("severity") == "medium"]),
            "low_count": len([v for v in scan_result.get("vulnerabilities", []) if v.get("severity") == "low"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan dependencies: {str(e)}")

@router.get("/dependencies/report")
async def get_dependency_report(
    current_user: Dict[str, Any] = Depends(get_current_user),
    format: str = Query("json", description="Report format: json, csv, html")
):
    """Get dependency vulnerability report"""
    try:
        dependency_scanner = get_dependency_scanner()
        if not dependency_scanner:
            raise HTTPException(status_code=503, detail="Dependency scanner not available")
        
        # Check cache
        cache_key = get_cache_key("dependency_report", {"format": format})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        report = dependency_scanner.generate_report(format)
        
        result = {
            "success": True,
            "format": format,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dependency report: {str(e)}")

# Secrets Management Operations
@router.post("/secrets/store")
async def store_secret(
    key: str = Form(...),
    value: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Store a secret (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        secrets_manager = get_secrets_manager()
        if not secrets_manager:
            raise HTTPException(status_code=503, detail="Secrets manager not available")
        
        secret_id = secrets_manager.store_secret(key, value, description)
        
        return {
            "success": True,
            "secret_id": secret_id,
            "key": key,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store secret: {str(e)}")

@router.get("/secrets/{secret_id}")
async def get_secret(
    secret_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a secret (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        secrets_manager = get_secrets_manager()
        if not secrets_manager:
            raise HTTPException(status_code=503, detail="Secrets manager not available")
        
        secret = secrets_manager.get_secret(secret_id)
        if not secret:
            raise HTTPException(status_code=404, detail="Secret not found")
        
        return {
            "success": True,
            "secret": secret,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get secret: {str(e)}")

@router.get("/secrets/list")
async def list_secrets(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = Query(100, description="Maximum number of secrets to return"),
    offset: int = Query(0, description="Number of secrets to skip")
):
    """List all secrets (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        secrets_manager = get_secrets_manager()
        if not secrets_manager:
            raise HTTPException(status_code=503, detail="Secrets manager not available")
        
        # Check cache
        cache_key = get_cache_key("list_secrets", {"limit": limit, "offset": offset})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        secrets = secrets_manager.list_secrets(limit=limit, offset=offset)
        
        result = {
            "success": True,
            "secrets": secrets,
            "total_count": len(secrets),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list secrets: {str(e)}")

@router.delete("/secrets/{secret_id}")
async def delete_secret(
    secret_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a secret (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        secrets_manager = get_secrets_manager()
        if not secrets_manager:
            raise HTTPException(status_code=503, detail="Secrets manager not available")
        
        success = secrets_manager.delete_secret(secret_id)
        
        return {
            "success": True,
            "deleted": success,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete secret: {str(e)}")

# OAuth Operations
@router.get("/oauth/providers")
async def get_oauth_providers():
    """Get available OAuth providers"""
    try:
        oauth_provider = get_oauth_provider()
        if not oauth_provider:
            raise HTTPException(status_code=503, detail="OAuth provider not available")
        
        providers = oauth_provider.get_available_providers()
        
        return {
            "success": True,
            "providers": providers,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get OAuth providers: {str(e)}")

@router.post("/oauth/authorize")
async def oauth_authorize(
    provider: str = Form(...),
    redirect_uri: str = Form(...)
):
    """Get OAuth authorization URL"""
    try:
        oauth_provider = get_oauth_provider()
        if not oauth_provider:
            raise HTTPException(status_code=503, detail="OAuth provider not available")
        
        auth_url = oauth_provider.get_authorization_url(provider, redirect_uri)
        
        return {
            "success": True,
            "provider": provider,
            "authorization_url": auth_url,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get OAuth authorization URL: {str(e)}")

@router.post("/oauth/callback")
async def oauth_callback(
    provider: str = Form(...),
    code: str = Form(...),
    state: Optional[str] = Form(None)
):
    """Handle OAuth callback"""
    try:
        oauth_provider = get_oauth_provider()
        if not oauth_provider:
            raise HTTPException(status_code=503, detail="OAuth provider not available")
        
        user_info = oauth_provider.handle_callback(provider, code, state)
        
        return {
            "success": True,
            "provider": provider,
            "user_info": user_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to handle OAuth callback: {str(e)}")

# SAML Operations
@router.get("/saml/metadata")
async def get_saml_metadata():
    """Get SAML metadata"""
    try:
        saml_provider = get_saml_provider()
        if not saml_provider:
            raise HTTPException(status_code=503, detail="SAML provider not available")
        
        metadata = saml_provider.get_metadata()
        
        return {
            "success": True,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SAML metadata: {str(e)}")

@router.post("/saml/login")
async def saml_login(
    saml_response: str = Form(...)
):
    """Handle SAML login"""
    try:
        saml_provider = get_saml_provider()
        if not saml_provider:
            raise HTTPException(status_code=503, detail="SAML provider not available")
        
        user_info = saml_provider.process_response(saml_response)
        
        return {
            "success": True,
            "user_info": user_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process SAML login: {str(e)}")

# Batch Operations
@router.post("/batch/security-scan")
async def batch_security_scan(
    scan_config: str = Form(..., description="JSON scan configuration"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Run batch security scans"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Parse scan configuration
        config = json.loads(scan_config)
        
        security_monitor = get_security_monitor()
        dependency_scanner = get_dependency_scanner()
        
        if not security_monitor or not dependency_scanner:
            raise HTTPException(status_code=503, detail="Security components not available")
        
        results = []
        
        # Run different types of scans
        for scan_type in config.get("scan_types", []):
            try:
                if scan_type == "dependency":
                    result = dependency_scanner.scan_project(config.get("project_path", "."))
                else:
                    result = security_monitor.run_scan(scan_type, config.get("target"))
                
                results.append({
                    "scan_type": scan_type,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "scan_type": scan_type,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "scan_config": config,
            "results": results,
            "total_scans": len(results),
            "successful_scans": len([r for r in results if r["status"] == "success"]),
            "failed_scans": len([r for r in results if r["status"] == "error"]),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run batch security scan: {str(e)}")

# Cache Management
@router.post("/cache/clear")
async def clear_cache(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Clear all cache entries (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        global _cache, _cache_timestamps
        cache_count = len(_cache)
        _cache.clear()
        _cache_timestamps.clear()
        
        return {
            "success": True,
            "cleared_entries": cache_count,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/cache/status")
async def get_cache_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get cache status and statistics (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        return {
            "success": True,
            "cache_entries": len(_cache),
            "cache_size_mb": sum(len(pickle.dumps(v)) for v in _cache.values()) / (1024 * 1024),
            "oldest_entry": min(_cache_timestamps.values()) if _cache_timestamps else None,
            "newest_entry": max(_cache_timestamps.values()) if _cache_timestamps else None,
            "rate_limit_entries": len(rate_limit_store),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

# System Information
@router.get("/system/info")
async def get_system_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get security system information (admin only)"""
    try:
        # Check admin role
        if "admin" not in current_user["roles"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        return {
            "success": True,
            "security_available": SECURITY_AVAILABLE,
            "auth_manager_ready": get_auth_manager() is not None,
            "security_monitor_ready": get_security_monitor() is not None,
            "dependency_scanner_ready": get_dependency_scanner() is not None,
            "secrets_manager_ready": get_secrets_manager() is not None,
            "oauth_provider_ready": get_oauth_provider() is not None,
            "saml_provider_ready": get_saml_provider() is not None,
            "cache_entries": len(_cache),
            "rate_limit_entries": len(rate_limit_store),
            "supported_features": [
                "authentication", "authorization", "user_management",
                "security_monitoring", "dependency_scanning", "secrets_management",
                "oauth_integration", "saml_integration", "rate_limiting"
            ],
            "security_config": {
                "jwt_expiry_hours": SECURITY_CONFIG["jwt_expiry_hours"],
                "max_login_attempts": SECURITY_CONFIG["max_login_attempts"],
                "password_min_length": SECURITY_CONFIG["password_min_length"],
                "session_timeout_minutes": SECURITY_CONFIG["session_timeout_minutes"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}") 