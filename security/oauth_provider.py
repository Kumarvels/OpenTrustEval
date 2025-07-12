"""
OAuth Provider for OpenTrustEval
Supports OAuth 2.0 with multiple providers (Google, GitHub, Microsoft, etc.)
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from urllib.parse import urlencode, urlparse, parse_qs

# Optional OAuth imports
try:
    import requests
    from authlib.integrations.starlette_client import OAuth
    from starlette.requests import Request
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

class OAuthProvider:
    """
    OAuth 2.0 provider supporting multiple services
    """
    
    def __init__(self):
        self.providers: Dict[str, Dict] = {}
        self.logger = logging.getLogger('OAuthProvider')
        
        # Load OAuth configurations
        self._load_oauth_configs()
    
    def _load_oauth_configs(self):
        """Load OAuth provider configurations"""
        # Google OAuth
        self.providers['google'] = {
            'client_id': os.getenv('GOOGLE_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
            'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
            'token_url': 'https://oauth2.googleapis.com/token',
            'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
            'scope': 'openid email profile'
        }
        
        # GitHub OAuth
        self.providers['github'] = {
            'client_id': os.getenv('GITHUB_CLIENT_ID'),
            'client_secret': os.getenv('GITHUB_CLIENT_SECRET'),
            'authorize_url': 'https://github.com/login/oauth/authorize',
            'token_url': 'https://github.com/login/oauth/access_token',
            'userinfo_url': 'https://api.github.com/user',
            'scope': 'read:user user:email'
        }
        
        # Microsoft OAuth
        self.providers['microsoft'] = {
            'client_id': os.getenv('MICROSOFT_CLIENT_ID'),
            'client_secret': os.getenv('MICROSOFT_CLIENT_SECRET'),
            'authorize_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
            'token_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
            'userinfo_url': 'https://graph.microsoft.com/v1.0/me',
            'scope': 'openid email profile'
        }
        
        # Azure AD OAuth
        self.providers['azure'] = {
            'client_id': os.getenv('AZURE_CLIENT_ID'),
            'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
            'tenant_id': os.getenv('AZURE_TENANT_ID'),
            'authorize_url': f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/v2.0/authorize",
            'token_url': f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/v2.0/token",
            'userinfo_url': 'https://graph.microsoft.com/v1.0/me',
            'scope': 'openid email profile'
        }
    
    def get_authorization_url(self, provider: str, redirect_uri: str, state: Optional[str] = None) -> str:
        """Generate OAuth authorization URL"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        params = {
            'client_id': config['client_id'],
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': config['scope']
        }
        
        if state:
            params['state'] = state
            
        if provider == 'google':
            params['access_type'] = 'offline'
            params['prompt'] = 'consent'
            
        query_string = urlencode(params)
        return f"{config['authorize_url']}?{query_string}"
    
    def exchange_code_for_token(self, provider: str, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        token_data = {
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'code': code,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.post(config['token_url'], data=token_data, headers=headers)
            response.raise_for_status()
            
            token_info = response.json()
            self.logger.info(f"Token exchange successful for {provider}")
            return token_info
            
        except requests.RequestException as e:
            self.logger.error(f"Token exchange failed for {provider}: {e}")
            raise
    
    def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(config['userinfo_url'], headers=headers)
            response.raise_for_status()
            
            user_info = response.json()
            
            # Normalize user info across providers
            normalized_info = self._normalize_user_info(provider, user_info)
            
            self.logger.info(f"User info retrieved for {provider}")
            return normalized_info
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get user info from {provider}: {e}")
            raise
    
    def _normalize_user_info(self, provider: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize user information across different providers"""
        normalized = {
            'provider': provider,
            'provider_user_id': None,
            'email': None,
            'name': None,
            'given_name': None,
            'family_name': None,
            'picture': None
        }
        
        if provider == 'google':
            normalized.update({
                'provider_user_id': user_info.get('id'),
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'given_name': user_info.get('given_name'),
                'family_name': user_info.get('family_name'),
                'picture': user_info.get('picture')
            })
            
        elif provider == 'github':
            normalized.update({
                'provider_user_id': str(user_info.get('id')),
                'email': user_info.get('email'),
                'name': user_info.get('name') or user_info.get('login'),
                'given_name': user_info.get('name', '').split()[0] if user_info.get('name') else None,
                'family_name': ' '.join(user_info.get('name', '').split()[1:]) if user_info.get('name') else None,
                'picture': user_info.get('avatar_url')
            })
            
        elif provider in ['microsoft', 'azure']:
            normalized.update({
                'provider_user_id': user_info.get('id'),
                'email': user_info.get('userPrincipalName') or user_info.get('mail'),
                'name': user_info.get('displayName'),
                'given_name': user_info.get('givenName'),
                'family_name': user_info.get('surname'),
                'picture': None  # Microsoft Graph doesn't provide picture in basic profile
            })
        
        return normalized
    
    def validate_token(self, provider: str, access_token: str) -> bool:
        """Validate OAuth access token"""
        try:
            user_info = self.get_user_info(provider, access_token)
            return user_info is not None
        except Exception:
            return False
    
    def refresh_token(self, provider: str, refresh_token: str) -> Dict[str, Any]:
        """Refresh OAuth access token"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        token_data = {
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.post(config['token_url'], data=token_data, headers=headers)
            response.raise_for_status()
            
            token_info = response.json()
            self.logger.info(f"Token refresh successful for {provider}")
            return token_info
            
        except requests.RequestException as e:
            self.logger.error(f"Token refresh failed for {provider}: {e}")
            raise
    
    def revoke_token(self, provider: str, access_token: str) -> bool:
        """Revoke OAuth access token"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        # Not all providers support token revocation
        if provider == 'google':
            revoke_url = 'https://oauth2.googleapis.com/revoke'
            data = {'token': access_token}
            
            try:
                response = requests.post(revoke_url, data=data)
                response.raise_for_status()
                self.logger.info(f"Token revoked for {provider}")
                return True
            except requests.RequestException as e:
                self.logger.error(f"Token revocation failed for {provider}: {e}")
                return False
                
        elif provider == 'github':
            # GitHub doesn't have a token revocation endpoint
            self.logger.warning(f"Token revocation not supported for {provider}")
            return False
            
        elif provider in ['microsoft', 'azure']:
            # Microsoft Graph doesn't have a token revocation endpoint
            self.logger.warning(f"Token revocation not supported for {provider}")
            return False
            
        return False
    
    def get_supported_providers(self) -> list:
        """Get list of supported OAuth providers"""
        return list(self.providers.keys())
    
    def is_provider_configured(self, provider: str) -> bool:
        """Check if OAuth provider is properly configured"""
        if provider not in self.providers:
            return False
            
        config = self.providers[provider]
        return bool(config.get('client_id') and config.get('client_secret'))
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get OAuth provider configuration (without secrets)"""
        if provider not in self.providers:
            raise ValueError(f"OAuth provider '{provider}' not supported")
            
        config = self.providers[provider].copy()
        # Remove sensitive information
        config.pop('client_secret', None)
        return config 