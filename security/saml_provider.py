"""
SAML Provider for OpenTrustEval
Supports SAML 2.0 for enterprise SSO integration
"""

import os
import base64
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Any, List
from urllib.parse import urlencode, urlparse, parse_qs
from datetime import datetime, timedelta

# Optional SAML imports
try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    from onelogin.saml2.utils import OneLogin_Saml2_Utils
    from onelogin.saml2.settings import OneLogin_Saml2_Settings
    SAML_AVAILABLE = True
except ImportError:
    SAML_AVAILABLE = False

class SAMLProvider:
    """
    SAML 2.0 provider for enterprise SSO
    """
    
    def __init__(self):
        self.providers: Dict[str, Dict] = {}
        self.logger = logging.getLogger('SAMLProvider')
        
        # Load SAML configurations
        self._load_saml_configs()
    
    def _load_saml_configs(self):
        """Load SAML provider configurations"""
        # Azure AD SAML
        self.providers['azure'] = {
            'entity_id': os.getenv('AZURE_SAML_ENTITY_ID'),
            'acs_url': os.getenv('AZURE_SAML_ACS_URL'),
            'slo_url': os.getenv('AZURE_SAML_SLO_URL'),
            'idp_entity_id': os.getenv('AZURE_SAML_IDP_ENTITY_ID'),
            'sso_url': os.getenv('AZURE_SAML_SSO_URL'),
            'idp_slo_url': os.getenv('AZURE_SAML_IDP_SLO_URL'),
            'x509cert': os.getenv('AZURE_SAML_X509CERT'),
            'name_id_format': 'urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified'
        }
        
        # Okta SAML
        self.providers['okta'] = {
            'entity_id': os.getenv('OKTA_SAML_ENTITY_ID'),
            'acs_url': os.getenv('OKTA_SAML_ACS_URL'),
            'slo_url': os.getenv('OKTA_SAML_SLO_URL'),
            'idp_entity_id': os.getenv('OKTA_SAML_IDP_ENTITY_ID'),
            'sso_url': os.getenv('OKTA_SAML_SSO_URL'),
            'idp_slo_url': os.getenv('OKTA_SAML_IDP_SLO_URL'),
            'x509cert': os.getenv('OKTA_SAML_X509CERT'),
            'name_id_format': 'urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress'
        }
        
        # ADFS SAML
        self.providers['adfs'] = {
            'entity_id': os.getenv('ADFS_SAML_ENTITY_ID'),
            'acs_url': os.getenv('ADFS_SAML_ACS_URL'),
            'slo_url': os.getenv('ADFS_SAML_SLO_URL'),
            'idp_entity_id': os.getenv('ADFS_SAML_IDP_ENTITY_ID'),
            'sso_url': os.getenv('ADFS_SAML_SSO_URL'),
            'idp_slo_url': os.getenv('ADFS_SAML_IDP_SLO_URL'),
            'x509cert': os.getenv('ADFS_SAML_X509CERT'),
            'name_id_format': 'urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified'
        }
    
    def get_saml_settings(self, provider: str) -> Dict[str, Any]:
        """Get SAML settings for provider"""
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        settings = {
            'strict': True,
            'debug': True,
            'sp': {
                'entityId': config['entity_id'],
                'assertionConsumerService': {
                    'url': config['acs_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'
                },
                'singleLogoutService': {
                    'url': config['slo_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'NameIDFormat': config['name_id_format']
            },
            'idp': {
                'entityId': config['idp_entity_id'],
                'singleSignOnService': {
                    'url': config['sso_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'singleLogoutService': {
                    'url': config['idp_slo_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'x509cert': config['x509cert']
            }
        }
        
        return settings
    
    def get_login_url(self, provider: str, relay_state: Optional[str] = None) -> str:
        """Generate SAML login URL"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        settings = self.get_saml_settings(provider)
        saml_settings = OneLogin_Saml2_Settings(settings)
        
        auth = OneLogin_Saml2_Auth({
            'http_host': 'localhost',
            'script_name': '/',
            'https': 'off'
        }, settings)
        
        login_url = auth.login(relay_state)
        self.logger.info(f"SAML login URL generated for {provider}")
        return login_url
    
    def process_response(self, provider: str, saml_response: str, relay_state: Optional[str] = None) -> Dict[str, Any]:
        """Process SAML response"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        settings = self.get_saml_settings(provider)
        
        auth = OneLogin_Saml2_Auth({
            'http_host': 'localhost',
            'script_name': '/',
            'https': 'off'
        }, settings)
        
        auth.process_response()
        errors = auth.get_errors()
        
        if errors:
            error_msg = f"SAML errors: {errors}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not auth.is_authenticated():
            raise ValueError("SAML authentication failed")
        
        # Extract user attributes
        attributes = auth.get_attributes()
        name_id = auth.get_nameid()
        
        # Normalize user info
        user_info = self._normalize_user_info(provider, attributes, name_id)
        
        self.logger.info(f"SAML authentication successful for {provider}")
        return user_info
    
    def _normalize_user_info(self, provider: str, attributes: Dict[str, List[str]], name_id: str) -> Dict[str, Any]:
        """Normalize user information from SAML attributes"""
        normalized = {
            'provider': provider,
            'provider_user_id': name_id,
            'email': None,
            'name': None,
            'given_name': None,
            'family_name': None,
            'groups': [],
            'attributes': attributes
        }
        
        # Extract email
        email_attrs = ['email', 'mail', 'userPrincipalName', 'EmailAddress']
        for attr in email_attrs:
            if attr in attributes and attributes[attr]:
                normalized['email'] = attributes[attr][0]
                break
        
        # Extract name
        name_attrs = ['displayName', 'name', 'cn', 'fullName']
        for attr in name_attrs:
            if attr in attributes and attributes[attr]:
                normalized['name'] = attributes[attr][0]
                break
        
        # Extract given name
        given_name_attrs = ['givenName', 'firstName', 'first_name']
        for attr in given_name_attrs:
            if attr in attributes and attributes[attr]:
                normalized['given_name'] = attributes[attr][0]
                break
        
        # Extract family name
        family_name_attrs = ['surname', 'lastName', 'last_name', 'sn']
        for attr in family_name_attrs:
            if attr in attributes and attributes[attr]:
                normalized['family_name'] = attributes[attr][0]
                break
        
        # Extract groups
        group_attrs = ['groups', 'memberOf', 'roles', 'groupMembership']
        for attr in group_attrs:
            if attr in attributes and attributes[attr]:
                normalized['groups'] = attributes[attr]
                break
        
        return normalized
    
    def get_logout_url(self, provider: str, name_id: str, session_index: Optional[str] = None, relay_state: Optional[str] = None) -> str:
        """Generate SAML logout URL"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        settings = self.get_saml_settings(provider)
        
        auth = OneLogin_Saml2_Auth({
            'http_host': 'localhost',
            'script_name': '/',
            'https': 'off'
        }, settings)
        
        logout_url = auth.logout(name_id, session_index, relay_state)
        self.logger.info(f"SAML logout URL generated for {provider}")
        return logout_url
    
    def process_logout_response(self, provider: str, saml_response: str) -> bool:
        """Process SAML logout response"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        settings = self.get_saml_settings(provider)
        
        auth = OneLogin_Saml2_Auth({
            'http_host': 'localhost',
            'script_name': '/',
            'https': 'off'
        }, settings)
        
        auth.process_slo()
        errors = auth.get_errors()
        
        if errors:
            self.logger.error(f"SAML logout errors: {errors}")
            return False
        
        self.logger.info(f"SAML logout successful for {provider}")
        return True
    
    def get_metadata(self, provider: str) -> str:
        """Generate SAML metadata for service provider"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        settings = self.get_saml_settings(provider)
        saml_settings = OneLogin_Saml2_Settings(settings)
        
        metadata = saml_settings.get_sp_metadata()
        errors = saml_settings.validate_metadata(metadata)
        
        if errors:
            raise ValueError(f"SAML metadata errors: {errors}")
        
        return metadata
    
    def validate_metadata(self, metadata: str) -> List[str]:
        """Validate SAML metadata"""
        if not SAML_AVAILABLE:
            raise RuntimeError("SAML library not available")
            
        try:
            # Parse XML metadata
            root = ET.fromstring(metadata)
            
            # Basic validation
            errors = []
            
            # Check for required elements
            required_elements = [
                './/{urn:oasis:names:tc:SAML:2.0:metadata}EntityDescriptor',
                './/{urn:oasis:names:tc:SAML:2.0:metadata}SPSSODescriptor',
                './/{urn:oasis:names:tc:SAML:2.0:metadata}AssertionConsumerService'
            ]
            
            for element_path in required_elements:
                if not root.findall(element_path):
                    errors.append(f"Missing required element: {element_path}")
            
            return errors
            
        except ET.ParseError as e:
            return [f"Invalid XML: {e}"]
    
    def get_supported_providers(self) -> list:
        """Get list of supported SAML providers"""
        return list(self.providers.keys())
    
    def is_provider_configured(self, provider: str) -> bool:
        """Check if SAML provider is properly configured"""
        if provider not in self.providers:
            return False
            
        config = self.providers[provider]
        required_fields = ['entity_id', 'acs_url', 'idp_entity_id', 'sso_url', 'x509cert']
        
        return all(config.get(field) for field in required_fields)
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get SAML provider configuration (without certificates)"""
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        config = self.providers[provider].copy()
        # Remove sensitive information
        config.pop('x509cert', None)
        return config
    
    def create_test_metadata(self, provider: str, base_url: str) -> str:
        """Create test SAML metadata for development"""
        if provider not in self.providers:
            raise ValueError(f"SAML provider '{provider}' not supported")
            
        config = self.providers[provider]
        
        # Create test settings
        test_settings = {
            'strict': True,
            'debug': True,
            'sp': {
                'entityId': f"{base_url}/saml/metadata",
                'assertionConsumerService': {
                    'url': f"{base_url}/saml/acs",
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'
                },
                'singleLogoutService': {
                    'url': f"{base_url}/saml/slo",
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'NameIDFormat': config['name_id_format']
            },
            'idp': {
                'entityId': config['idp_entity_id'],
                'singleSignOnService': {
                    'url': config['sso_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'singleLogoutService': {
                    'url': config['idp_slo_url'],
                    'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
                },
                'x509cert': config['x509cert']
            }
        }
        
        if SAML_AVAILABLE:
            saml_settings = OneLogin_Saml2_Settings(test_settings)
            return saml_settings.get_sp_metadata()
        else:
            # Return basic XML structure if SAML library not available
            return f"""<?xml version="1.0"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" entityID="{base_url}/saml/metadata">
    <md:SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="{base_url}/saml/acs"/>
        <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" Location="{base_url}/saml/slo"/>
        <md:NameIDFormat>{config['name_id_format']}</md:NameIDFormat>
    </md:SPSSODescriptor>
</md:EntityDescriptor>""" 