"""
Security Module for OpenTrustEval
Comprehensive security framework including authentication, authorization, and security monitoring.
"""

from .auth_manager import AuthManager
from .oauth_provider import OAuthProvider
from .saml_provider import SAMLProvider
from .security_monitor import SecurityMonitor
from .dependency_scanner import DependencyScanner
from .secrets_manager import SecretsManager

__all__ = [
    'AuthManager',
    'OAuthProvider',
    'SAMLProvider',
    'SecurityMonitor',
    'DependencyScanner',
    'SecretsManager'
]