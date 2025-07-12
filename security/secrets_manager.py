"""
Secrets Manager for OpenTrustEval
Secure credential storage and management with encryption
"""

import os
import json
import base64
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac

# Optional encryption imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

class SecretType(Enum):
    """Secret types"""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    DATABASE = "database"
    OAUTH = "oauth"
    SAML = "saml"

@dataclass
class Secret:
    """Secret information"""
    id: str
    name: str
    secret_type: SecretType
    encrypted_value: str
    description: Optional[str] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    created_by: Optional[str] = None

class SecretsManager:
    """
    Secure secrets management with encryption
    """
    
    def __init__(self, master_key: Optional[str] = None, storage_path: Optional[str] = None):
        self.secrets: Dict[str, Secret] = {}
        self.master_key = master_key or os.getenv('SECRETS_MASTER_KEY')
        self.storage_path = storage_path or os.getenv('SECRETS_STORAGE_PATH', './secrets.json')
        self.fernet = None
        
        # Security settings
        self.max_secret_length = 4096
        self.key_rotation_days = 90
        self.audit_log: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = logging.getLogger('SecretsManager')
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load existing secrets
        self._load_secrets()
    
    def _initialize_encryption(self):
        """Initialize encryption with master key"""
        if not ENCRYPTION_AVAILABLE:
            self.logger.warning("Encryption not available - secrets will be stored in plain text")
            return
        
        if not self.master_key:
            # Generate a new master key
            self.master_key = base64.urlsafe_b64encode(Fernet.generate_key()).decode()
            self.logger.warning("No master key provided - generated new key. Store this securely!")
            self.logger.warning(f"Master key: {self.master_key}")
        
        try:
            # Derive encryption key from master key
            salt = b'opentrusteval_salt'  # In production, use a random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self.fernet = Fernet(key)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            self.fernet = None
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value"""
        if not self.fernet:
            # Fallback to base64 encoding if encryption not available
            return base64.b64encode(value.encode()).decode()
        
        try:
            encrypted = self.fernet.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value"""
        if not self.fernet:
            # Fallback to base64 decoding if encryption not available
            return base64.b64decode(encrypted_value.encode()).decode()
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def _load_secrets(self):
        """Load secrets from storage"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for secret_data in data.get('secrets', []):
                secret = Secret(
                    id=secret_data['id'],
                    name=secret_data['name'],
                    secret_type=SecretType(secret_data['secret_type']),
                    encrypted_value=secret_data['encrypted_value'],
                    description=secret_data.get('description'),
                    tags=secret_data.get('tags', []),
                    created_at=datetime.fromisoformat(secret_data['created_at']),
                    updated_at=datetime.fromisoformat(secret_data['updated_at']),
                    expires_at=datetime.fromisoformat(secret_data['expires_at']) if secret_data.get('expires_at') else None,
                    created_by=secret_data.get('created_by')
                )
                self.secrets[secret.id] = secret
            
            self.logger.info(f"Loaded {len(self.secrets)} secrets from storage")
            
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
    
    def _save_secrets(self):
        """Save secrets to storage"""
        try:
            data = {
                'secrets': [
                    {
                        'id': secret.id,
                        'name': secret.name,
                        'secret_type': secret.secret_type.value,
                        'encrypted_value': secret.encrypted_value,
                        'description': secret.description,
                        'tags': secret.tags,
                        'created_at': secret.created_at.isoformat(),
                        'updated_at': secret.updated_at.isoformat(),
                        'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                        'created_by': secret.created_by
                    }
                    for secret in self.secrets.values()
                ]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.secrets)} secrets to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
    
    def _log_audit(self, action: str, secret_id: str, user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'secret_id': secret_id,
            'user_id': user_id,
            'details': details or {}
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def create_secret(self, name: str, value: str, secret_type: SecretType, 
                     description: Optional[str] = None, tags: Optional[List[str]] = None,
                     expires_in_days: Optional[int] = None, created_by: Optional[str] = None) -> str:
        """Create a new secret"""
        # Validate input
        if not name or not value:
            raise ValueError("Name and value are required")
        
        if len(value) > self.max_secret_length:
            raise ValueError(f"Secret value too long (max {self.max_secret_length} characters)")
        
        # Check for duplicate names
        for secret in self.secrets.values():
            if secret.name == name:
                raise ValueError(f"Secret with name '{name}' already exists")
        
        # Create secret
        secret_id = secrets.token_urlsafe(16)
        encrypted_value = self._encrypt_value(value)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        secret = Secret(
            id=secret_id,
            name=name,
            secret_type=secret_type,
            encrypted_value=encrypted_value,
            description=description,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expires_at=expires_at,
            created_by=created_by
        )
        
        self.secrets[secret_id] = secret
        self._save_secrets()
        self._log_audit('create', secret_id, created_by, {'name': name, 'type': secret_type.value})
        
        self.logger.info(f"Secret created: {name}")
        return secret_id
    
    def get_secret(self, secret_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """Get secret value by ID"""
        if secret_id not in self.secrets:
            return None
        
        secret = self.secrets[secret_id]
        
        # Check if expired
        if secret.expires_at and datetime.now() > secret.expires_at:
            self.logger.warning(f"Secret expired: {secret.name}")
            return None
        
        try:
            value = self._decrypt_value(secret.encrypted_value)
            self._log_audit('read', secret_id, user_id, {'name': secret.name})
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt secret {secret.name}: {e}")
            return None
    
    def get_secret_by_name(self, name: str, user_id: Optional[str] = None) -> Optional[str]:
        """Get secret value by name"""
        for secret in self.secrets.values():
            if secret.name == name:
                return self.get_secret(secret.id, user_id)
        return None
    
    def update_secret(self, secret_id: str, value: str, description: Optional[str] = None,
                     tags: Optional[List[str]] = None, expires_in_days: Optional[int] = None,
                     updated_by: Optional[str] = None) -> bool:
        """Update a secret"""
        if secret_id not in self.secrets:
            return False
        
        secret = self.secrets[secret_id]
        
        # Validate input
        if len(value) > self.max_secret_length:
            raise ValueError(f"Secret value too long (max {self.max_secret_length} characters)")
        
        # Update secret
        secret.encrypted_value = self._encrypt_value(value)
        secret.updated_at = datetime.now()
        
        if description is not None:
            secret.description = description
        
        if tags is not None:
            secret.tags = tags
        
        if expires_in_days is not None:
            if expires_in_days == 0:
                secret.expires_at = None
            else:
                secret.expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        self._save_secrets()
        self._log_audit('update', secret_id, updated_by, {'name': secret.name})
        
        self.logger.info(f"Secret updated: {secret.name}")
        return True
    
    def delete_secret(self, secret_id: str, deleted_by: Optional[str] = None) -> bool:
        """Delete a secret"""
        if secret_id not in self.secrets:
            return False
        
        secret = self.secrets[secret_id]
        secret_name = secret.name
        
        del self.secrets[secret_id]
        self._save_secrets()
        self._log_audit('delete', secret_id, deleted_by, {'name': secret_name})
        
        self.logger.info(f"Secret deleted: {secret_name}")
        return True
    
    def list_secrets(self, secret_type: Optional[SecretType] = None, 
                    tags: Optional[List[str]] = None, include_expired: bool = False) -> List[Dict[str, Any]]:
        """List secrets with optional filtering"""
        secrets_list = []
        
        for secret in self.secrets.values():
            # Filter by type
            if secret_type and secret.secret_type != secret_type:
                continue
            
            # Filter by tags
            if tags and not all(tag in secret.tags for tag in tags):
                continue
            
            # Filter by expiration
            if not include_expired and secret.expires_at and datetime.now() > secret.expires_at:
                continue
            
            secrets_list.append({
                'id': secret.id,
                'name': secret.name,
                'type': secret.secret_type.value,
                'description': secret.description,
                'tags': secret.tags,
                'created_at': secret.created_at.isoformat(),
                'updated_at': secret.updated_at.isoformat(),
                'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                'expired': secret.expires_at and datetime.now() > secret.expires_at if secret.expires_at else False,
                'created_by': secret.created_by
            })
        
        return secrets_list
    
    def search_secrets(self, query: str) -> List[Dict[str, Any]]:
        """Search secrets by name, description, or tags"""
        results = []
        query_lower = query.lower()
        
        for secret in self.secrets.values():
            # Search in name
            if query_lower in secret.name.lower():
                results.append(self._secret_to_dict(secret))
                continue
            
            # Search in description
            if secret.description and query_lower in secret.description.lower():
                results.append(self._secret_to_dict(secret))
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in secret.tags):
                results.append(self._secret_to_dict(secret))
                continue
        
        return results
    
    def _secret_to_dict(self, secret: Secret) -> Dict[str, Any]:
        """Convert secret to dictionary (without value)"""
        return {
            'id': secret.id,
            'name': secret.name,
            'type': secret.secret_type.value,
            'description': secret.description,
            'tags': secret.tags,
            'created_at': secret.created_at.isoformat(),
            'updated_at': secret.updated_at.isoformat(),
            'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
            'expired': secret.expires_at and datetime.now() > secret.expires_at if secret.expires_at else False,
            'created_by': secret.created_by
        }
    
    def rotate_secret(self, secret_id: str, new_value: str, rotated_by: Optional[str] = None) -> bool:
        """Rotate a secret value"""
        if secret_id not in self.secrets:
            return False
        
        secret = self.secrets[secret_id]
        
        # Store old value for audit
        old_value = self._decrypt_value(secret.encrypted_value)
        
        # Update with new value
        success = self.update_secret(secret_id, new_value, updated_by=rotated_by)
        
        if success:
            self._log_audit('rotate', secret_id, rotated_by, {
                'name': secret.name,
                'old_value_hash': hashlib.sha256(old_value.encode()).hexdigest()[:8]
            })
        
        return success
    
    def get_expiring_secrets(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get secrets expiring within specified days"""
        cutoff_date = datetime.now() + timedelta(days=days)
        expiring_secrets = []
        
        for secret in self.secrets.values():
            if secret.expires_at and secret.expires_at <= cutoff_date:
                expiring_secrets.append(self._secret_to_dict(secret))
        
        return expiring_secrets
    
    def extend_secret_expiry(self, secret_id: str, additional_days: int, extended_by: Optional[str] = None) -> bool:
        """Extend secret expiration"""
        if secret_id not in self.secrets:
            return False
        
        secret = self.secrets[secret_id]
        
        if secret.expires_at:
            secret.expires_at += timedelta(days=additional_days)
        else:
            secret.expires_at = datetime.now() + timedelta(days=additional_days)
        
        secret.updated_at = datetime.now()
        self._save_secrets()
        self._log_audit('extend_expiry', secret_id, extended_by, {
            'name': secret.name,
            'additional_days': additional_days
        })
        
        self.logger.info(f"Secret expiry extended: {secret.name}")
        return True
    
    def get_audit_log(self, secret_id: Optional[str] = None, 
                     action: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log with optional filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_log = [
            entry for entry in self.audit_log
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if secret_id:
            filtered_log = [entry for entry in filtered_log if entry['secret_id'] == secret_id]
        
        if action:
            filtered_log = [entry for entry in filtered_log if entry['action'] == action]
        
        return filtered_log
    
    def export_secrets(self, format: str = 'json', include_values: bool = False, 
                      output_path: Optional[str] = None) -> str:
        """Export secrets"""
        if format.lower() == 'json':
            if include_values:
                export_data = []
                for secret in self.secrets.values():
                    secret_data = self._secret_to_dict(secret)
                    try:
                        secret_data['value'] = self._decrypt_value(secret.encrypted_value)
                    except Exception:
                        secret_data['value'] = '[ENCRYPTED]'
                    export_data.append(secret_data)
            else:
                export_data = [self._secret_to_dict(secret) for secret in self.secrets.values()]
            
            content = json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Secrets exported to {output_path}")
        
        return content
    
    def import_secrets(self, data: str, format: str = 'json', imported_by: Optional[str] = None) -> int:
        """Import secrets"""
        if format.lower() == 'json':
            secrets_data = json.loads(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        imported_count = 0
        
        for secret_data in secrets_data:
            try:
                # Create secret
                secret_id = self.create_secret(
                    name=secret_data['name'],
                    value=secret_data.get('value', ''),
                    secret_type=SecretType(secret_data['type']),
                    description=secret_data.get('description'),
                    tags=secret_data.get('tags', []),
                    created_by=imported_by
                )
                imported_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to import secret {secret_data.get('name', 'unknown')}: {e}")
        
        self.logger.info(f"Imported {imported_count} secrets")
        return imported_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get secrets statistics"""
        total_secrets = len(self.secrets)
        expired_secrets = len([s for s in self.secrets.values() if s.expires_at and datetime.now() > s.expires_at])
        expiring_soon = len(self.get_expiring_secrets(7))
        
        type_counts = {}
        for secret_type in SecretType:
            type_counts[secret_type.value] = len([s for s in self.secrets.values() if s.secret_type == secret_type])
        
        return {
            'total_secrets': total_secrets,
            'expired_secrets': expired_secrets,
            'expiring_soon': expiring_soon,
            'type_distribution': type_counts,
            'storage_size_mb': os.path.getsize(self.storage_path) / (1024 * 1024) if os.path.exists(self.storage_path) else 0
        } 