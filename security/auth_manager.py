"""
Authentication Manager for OpenTrustEval
Handles SSO, OAuth, SAML, and API key authentication with role-based access control.
"""

import os
import jwt
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json

# Optional imports for advanced auth providers
try:
    from authlib.integrations.starlette_client import OAuth
    from starlette.middleware.sessions import SessionMiddleware
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    from onelogin.saml2.utils import OneLogin_Saml2_Utils
    SAML_AVAILABLE = True
except ImportError:
    SAML_AVAILABLE = False

class AuthMethod(Enum):
    """Supported authentication methods"""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    SAML = "saml"
    SSO = "sso"

class UserRole(Enum):
    """User roles with permissions"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API_USER = "api_user"

@dataclass
class User:
    """User information"""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class AuthSession:
    """Authentication session"""
    session_id: str
    user_id: str
    method: AuthMethod
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

class AuthManager:
    """
    Central authentication manager supporting multiple auth methods
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.api_keys: Dict[str, str] = {}  # key_hash -> user_id
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        self.jwt_algorithm = 'HS256'
        self.jwt_expiry_hours = 24
        
        # OAuth and SAML providers
        self.oauth_providers: Dict[str, Any] = {}
        self.saml_providers: Dict[str, Any] = {}
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Logging
        self.logger = logging.getLogger('AuthManager')
        
        # Load configuration
        self._load_config(config_path)
        self._load_users()
        
    def _load_config(self, config_path: Optional[str]):
        """Load authentication configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load OAuth providers
            for provider_name, provider_config in config.get('oauth_providers', {}).items():
                if OAUTH_AVAILABLE:
                    self._setup_oauth_provider(provider_name, provider_config)
                    
            # Load SAML providers
            for provider_name, provider_config in config.get('saml_providers', {}).items():
                if SAML_AVAILABLE:
                    self._setup_saml_provider(provider_name, provider_config)
    
    def _setup_oauth_provider(self, name: str, config: Dict):
        """Setup OAuth provider"""
        if not OAUTH_AVAILABLE:
            self.logger.warning(f"OAuth not available, skipping {name}")
            return
            
        try:
            oauth = OAuth()
            oauth.register(
                name=name,
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                server_metadata_url=config.get('metadata_url'),
                client_kwargs=config.get('client_kwargs', {})
            )
            self.oauth_providers[name] = oauth
            self.logger.info(f"OAuth provider {name} configured")
        except Exception as e:
            self.logger.error(f"Failed to setup OAuth provider {name}: {e}")
    
    def _setup_saml_provider(self, name: str, config: Dict):
        """Setup SAML provider"""
        if not SAML_AVAILABLE:
            self.logger.warning(f"SAML not available, skipping {name}")
            return
            
        try:
            self.saml_providers[name] = config
            self.logger.info(f"SAML provider {name} configured")
        except Exception as e:
            self.logger.error(f"Failed to setup SAML provider {name}: {e}")
    
    def _load_users(self):
        """Load users from database or file"""
        # For demo purposes, create some default users
        default_users = [
            User(
                id="admin-001",
                username="admin",
                email="admin@opentrusteval.com",
                role=UserRole.ADMIN,
                permissions=["*"],
                created_at=datetime.now()
            ),
            User(
                id="user-001", 
                username="user",
                email="user@opentrusteval.com",
                role=UserRole.USER,
                permissions=["read", "write"],
                created_at=datetime.now()
            )
        ]
        
        for user in default_users:
            self.users[user.id] = user
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        if not api_key:
            return None
            
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_id = self.api_keys.get(key_hash)
        
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.is_active:
                self._log_auth_success(user.id, AuthMethod.API_KEY)
                return user
                
        self._log_auth_failure("api_key", "Invalid API key")
        return None
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate using JWT token"""
        if not token:
            return None
            
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get('user_id')
            
            if user_id and user_id in self.users:
                user = self.users[user_id]
                if user.is_active:
                    self._log_auth_success(user.id, AuthMethod.JWT)
                    return user
                    
        except jwt.ExpiredSignatureError:
            self._log_auth_failure("jwt", "Token expired")
        except jwt.InvalidTokenError:
            self._log_auth_failure("jwt", "Invalid token")
            
        return None
    
    def authenticate_oauth(self, provider_name: str, code: str, redirect_uri: str) -> Optional[User]:
        """Authenticate using OAuth"""
        if not OAUTH_AVAILABLE:
            self.logger.error("OAuth not available")
            return None
            
        if provider_name not in self.oauth_providers:
            self.logger.error(f"OAuth provider {provider_name} not found")
            return None
            
        try:
            oauth = self.oauth_providers[provider_name]
            token = oauth.fetch_token(code=code, redirect_uri=redirect_uri)
            user_info = oauth.parse_id_token(token)
            
            # Find or create user based on OAuth info
            user = self._get_or_create_oauth_user(user_info)
            if user:
                self._log_auth_success(user.id, AuthMethod.OAUTH)
                return user
                
        except Exception as e:
            self.logger.error(f"OAuth authentication failed: {e}")
            self._log_auth_failure("oauth", str(e))
            
        return None
    
    def authenticate_saml(self, provider_name: str, saml_response: str) -> Optional[User]:
        """Authenticate using SAML"""
        if not SAML_AVAILABLE:
            self.logger.error("SAML not available")
            return None
            
        if provider_name not in self.saml_providers:
            self.logger.error(f"SAML provider {provider_name} not found")
            return None
            
        try:
            # Process SAML response
            auth = OneLogin_Saml2_Auth({
                'http_host': 'localhost',
                'script_name': '/',
                'https': 'off'
            }, {
                'strict': True,
                'debug': True,
                'sp': {
                    'entityId': self.saml_providers[provider_name]['entity_id'],
                    'assertionConsumerService': {
                        'url': self.saml_providers[provider_name]['acs_url'],
                        'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'
                    },
                    'singleLogoutService': {
                        'url': self.saml_providers[provider_name]['slo_url'],
                        'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                    },
                    'NameIDFormat': 'urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified'
                },
                'idp': {
                    'entityId': self.saml_providers[provider_name]['idp_entity_id'],
                    'singleSignOnService': {
                        'url': self.saml_providers[provider_name]['sso_url'],
                        'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                    },
                    'singleLogoutService': {
                        'url': self.saml_providers[provider_name]['idp_slo_url'],
                        'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                    },
                    'x509cert': self.saml_providers[provider_name]['x509cert']
                }
            })
            
            auth.process_response()
            errors = auth.get_errors()
            
            if not errors:
                user_info = auth.get_attributes()
                user = self._get_or_create_saml_user(user_info)
                if user:
                    self._log_auth_success(user.id, AuthMethod.SAML)
                    return user
            else:
                self.logger.error(f"SAML errors: {errors}")
                self._log_auth_failure("saml", f"Errors: {errors}")
                
        except Exception as e:
            self.logger.error(f"SAML authentication failed: {e}")
            self._log_auth_failure("saml", str(e))
            
        return None
    
    def _get_or_create_oauth_user(self, user_info: Dict) -> Optional[User]:
        """Get or create user from OAuth info"""
        email = user_info.get('email')
        if not email:
            return None
            
        # Check if user exists
        for user in self.users.values():
            if user.email == email:
                return user
                
        # Create new user
        user_id = f"oauth-{secrets.token_hex(8)}"
        user = User(
            id=user_id,
            username=user_info.get('name', email.split('@')[0]),
            email=email,
            role=UserRole.USER,
            permissions=["read", "write"],
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        return user
    
    def _get_or_create_saml_user(self, user_info: Dict) -> Optional[User]:
        """Get or create user from SAML info"""
        email = user_info.get('email', [None])[0]
        if not email:
            return None
            
        # Check if user exists
        for user in self.users.values():
            if user.email == email:
                return user
                
        # Create new user
        user_id = f"saml-{secrets.token_hex(8)}"
        user = User(
            id=user_id,
            username=user_info.get('name', [email.split('@')[0]])[0],
            email=email,
            role=UserRole.USER,
            permissions=["read", "write"],
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        return user
    
    def create_session(self, user: User, method: AuthMethod, ip_address: str, user_agent: str) -> str:
        """Create authentication session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=self.jwt_expiry_hours)
        
        session = AuthSession(
            session_id=session_id,
            user_id=user.id,
            method=method,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        user.last_login = datetime.now()
        
        self.logger.info(f"Session created for user {user.username}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        if datetime.now() > session.expires_at:
            del self.sessions[session_id]
            return None
            
        user = self.users.get(session.user_id)
        if user and user.is_active:
            return user
            
        return None
    
    def create_jwt_token(self, user: User) -> str:
        """Create JWT token for user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[key_hash] = user.id
        
        self.logger.info(f"API key generated for user {user.username}")
        return api_key
    
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        if user.role == UserRole.ADMIN:
            return True
            
        if "*" in user.permissions:
            return True
            
        return permission in user.permissions
    
    def _log_auth_success(self, user_id: str, method: AuthMethod):
        """Log successful authentication"""
        self.logger.info(f"Authentication successful: user={user_id}, method={method.value}")
        # Clear failed attempts
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
    
    def _log_auth_failure(self, method: str, reason: str):
        """Log failed authentication"""
        self.logger.warning(f"Authentication failed: method={method}, reason={reason}")
    
    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        if identifier not in self.failed_attempts:
            return False
            
        attempts = self.failed_attempts[identifier]
        cutoff_time = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
        
        # Remove old attempts
        attempts = [attempt for attempt in attempts if attempt > cutoff_time]
        self.failed_attempts[identifier] = attempts
        
        return len(attempts) >= self.max_failed_attempts
    
    def record_failed_attempt(self, identifier: str):
        """Record failed authentication attempt"""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
            
        self.failed_attempts[identifier].append(datetime.now())
    
    def logout(self, session_id: str):
        """Logout user by invalidating session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"User logged out: session={session_id}")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def list_users(self) -> List[User]:
        """List all users"""
        return list(self.users.values())
    
    def create_user(self, username: str, email: str, role: UserRole, permissions: List[str]) -> User:
        """Create new user"""
        user_id = f"user-{secrets.token_hex(8)}"
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        self.logger.info(f"User created: {username}")
        return user
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user information"""
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        self.logger.info(f"User updated: {user_id}")
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if user_id not in self.users:
            return False
            
        del self.users[user_id]
        
        # Remove user's API keys
        keys_to_remove = [key for key, uid in self.api_keys.items() if uid == user_id]
        for key in keys_to_remove:
            del self.api_keys[key]
            
        # Remove user's sessions
        sessions_to_remove = [sid for sid, session in self.sessions.items() if session.user_id == user_id]
        for sid in sessions_to_remove:
            del self.sessions[sid]
            
        self.logger.info(f"User deleted: {user_id}")
        return True 