"""
Security Module for OpenTrustEval
Comprehensive security framework including authentication, authorization, and security monitoring.
"""

from src.opentrusteval.security.auth_manager import AuthManager
from src.opentrusteval.security.oauth_provider import OAuthProvider
from src.opentrusteval.security.saml_provider import SAMLProvider
from src.opentrusteval.security.security_monitor import SecurityMonitor
from src.opentrusteval.security.dependency_scanner import DependencyScanner
from src.opentrusteval.security.secrets_manager import SecretsManager

__all__ = [
    'AuthManager',
    'OAuthProvider',
    'SAMLProvider',
    'SecurityMonitor',
    'DependencyScanner',
    'SecretsManager'
]