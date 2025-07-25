# 🔒 OpenTrustEval Security Framework

Comprehensive security framework providing authentication, authorization, monitoring, and compliance features for OpenTrustEval.

## 🚀 Quick Start

### Installation

```bash
# Install security dependencies
pip install -r requirements_security.txt

# Set up environment variables
cp security/security_config.yaml.example security/security_config.yaml
# Edit the configuration file with your settings
```

### Basic Usage

```python
from security.auth_manager import AuthManager
from security.security_monitor import SecurityMonitor
from security.secrets_manager import SecretsManager

# Initialize security components
auth_manager = AuthManager()
security_monitor = SecurityMonitor()
secrets_manager = SecretsManager()

# Create a user
user = auth_manager.create_user("admin", "admin@example.com", UserRole.ADMIN, ["*"])

# Monitor security events
security_monitor.record_authentication_success("192.168.1.1", user.id)

# Store secrets securely
secret_id = secrets_manager.create_secret("api_key", "your-secret-key", SecretType.API_KEY)
```

## 🏗️ Architecture

### Core Components

1. **Authentication Manager** (`auth_manager.py`)
   - Multi-provider authentication (API keys, JWT, OAuth, SAML)
   - Role-based access control (RBAC)
   - Session management
   - Rate limiting and account lockout

2. **OAuth Provider** (`oauth_provider.py`)
   - Google, GitHub, Microsoft, Azure AD integration
   - Token management and validation
   - User information normalization

3. **SAML Provider** (`saml_provider.py`)
   - Enterprise SSO integration
   - Azure AD, Okta, ADFS support
   - Metadata generation and validation

4. **Security Monitor** (`security_monitor.py`)
   - Real-time threat detection
   - IP blocking and rate limiting
   - Security event logging and alerting
   - Incident response automation

5. **Dependency Scanner** (`dependency_scanner.py`)
   - Vulnerability scanning
   - License compliance checking
   - Security report generation

6. **Secrets Manager** (`secrets_manager.py`)
   - Encrypted credential storage
   - Key rotation and expiration
   - Audit logging

## 🔐 Authentication Methods

### 1. API Key Authentication

```python
# Generate API key
api_key = auth_manager.generate_api_key(user)

# Authenticate with API key
user = auth_manager.authenticate_api_key(api_key)
```

### 2. JWT Authentication

```python
# Create JWT token
token = auth_manager.create_jwt_token(user)

# Authenticate with JWT
user = auth_manager.authenticate_jwt(token)
```

### 3. OAuth Authentication

```python
# Get OAuth authorization URL
auth_url = oauth_provider.get_authorization_url("google", redirect_uri)

# Exchange code for token
token_info = oauth_provider.exchange_code_for_token("google", code, redirect_uri)

# Get user information
user_info = oauth_provider.get_user_info("google", token_info["access_token"])
```

### 4. SAML Authentication

```python
# Get SAML login URL
login_url = saml_provider.get_login_url("azure")

# Process SAML response
user_info = saml_provider.process_response("azure", saml_response)
```

## 🛡️ Security Monitoring

### Event Recording

```python
# Record authentication events
security_monitor.record_authentication_success("192.168.1.1", user_id)
security_monitor.record_authentication_failure("192.168.1.1", user_id)

# Record admin access
security_monitor.record_admin_access("192.168.1.1", user_id, "config_update")

# Record data access
security_monitor.record_data_access("192.168.1.1", user_id, "sensitive_data")
```

### Threat Detection

```python
# Get security summary
summary = security_monitor.get_security_summary()

# Get recent alerts
alerts = security_monitor.get_recent_alerts(hours=24)

# Get blocked IPs
blocked_ips = security_monitor.get_blocked_ips()
```

### IP Management

```python
# Block suspicious IP
security_monitor.block_ip("192.168.1.100", "Multiple failed login attempts")

# Check if IP is blocked
is_blocked = security_monitor.is_ip_blocked("192.168.1.100")

# Unblock IP
security_monitor.unblock_ip("192.168.1.100")
```

## 🔍 Dependency Scanning

### Vulnerability Scanning

```python
# Scan requirements file
results = dependency_scanner.scan_requirements_file("requirements.txt")

# Scan installed packages
results = dependency_scanner.scan_installed_packages()

# Generate report
report = dependency_scanner.generate_report(results, "security_report.md")
```

### License Compliance

```python
# Check license compliance
allowed_licenses = ["MIT", "Apache-2.0", "BSD-3-Clause"]
compliance = dependency_scanner.check_license_compliance("requirements.txt", allowed_licenses)
```

## 🔐 Secrets Management

### Secret Operations

```python
# Create secret
secret_id = secrets_manager.create_secret(
    "database_password",
    "secure_password_123",
    SecretType.PASSWORD,
    description="Database connection password",
    tags=["database", "production"],
    expires_in_days=90
)

# Get secret value
password = secrets_manager.get_secret_by_name("database_password")

# Update secret
secrets_manager.update_secret(secret_id, "new_password_456")

# Delete secret
secrets_manager.delete_secret("database_password")
```

### Secret Statistics

```python
# Get statistics
stats = secrets_manager.get_statistics()

# List expiring secrets
expiring = secrets_manager.get_expiring_secrets(days=30)
```

## 🖥️ Command Line Interface

### Installation

```bash
# Make CLI executable
chmod +x security/security_cli.py
```

### Usage Examples

```bash
# User management
python security_cli.py auth user create --username admin --email admin@example.com --role admin
python security_cli.py auth user list
python security_cli.py auth user delete --user-id user-123

# OAuth management
python security_cli.py auth oauth list
python security_cli.py auth oauth test --provider google

# SAML management
python security_cli.py auth saml list
python security_cli.py auth saml metadata --provider azure --base-url https://example.com

# Security monitoring
python security_cli.py monitor summary
python security_cli.py monitor alerts --hours 24 --severity high
python security_cli.py monitor blocks
python security_cli.py monitor unblock --ip 192.168.1.100

# Dependency scanning
python security_cli.py scan dependencies --requirements requirements.txt --output results.json
python security_cli.py scan installed --output results.json
python security_cli.py scan licenses --requirements requirements.txt --allowed "MIT,Apache-2.0"

# Secrets management
python security_cli.py secrets create --name api_key --value secret123 --type api_key
python security_cli.py secrets list --type api_key
python security_cli.py secrets get --name api_key
python security_cli.py secrets delete --name api_key
python security_cli.py secrets export --output secrets.json --include-values

# Configuration
python security_cli.py config show
python security_cli.py config validate --config security/security_config.yaml
```

## 🌐 Web Interface

### Launch WebUI

```bash
python security/security_webui.py
```

The WebUI will be available at `http://localhost:7862`

### Features

- **Authentication Management**: User creation, OAuth/SAML provider configuration
- **Security Monitoring**: Real-time alerts, IP management, threat detection
- **Dependency Scanning**: Vulnerability scanning, license compliance
- **Secrets Management**: Secure credential storage and management
- **Configuration**: Security settings management and validation

## ⚙️ Configuration

### Environment Variables

```bash
# JWT Configuration
export JWT_SECRET="your-super-secret-jwt-key"

# OAuth Configuration
export GOOGLE_CLIENT_ID="your-google-client-id"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
export GITHUB_CLIENT_ID="your-github-client-id"
export GITHUB_CLIENT_SECRET="your-github-client-secret"

# SAML Configuration
export AZURE_SAML_ENTITY_ID="your-entity-id"
export AZURE_SAML_ACS_URL="https://your-domain/saml/acs"
export AZURE_SAML_X509CERT="your-x509-certificate"

# Secrets Management
export SECRETS_MASTER_KEY="your-master-key"
export SECRETS_STORAGE_PATH="./secrets.json"
```

### Configuration File

The main configuration file is `security/security_config.yaml`. Key sections:

- **Authentication**: JWT settings, password policies
- **OAuth Providers**: Client IDs, secrets, endpoints
- **SAML Providers**: Entity IDs, certificates, endpoints
- **Security Monitoring**: Thresholds, rate limits, alerting
- **Secrets Management**: Encryption settings, key rotation
- **Dependency Scanning**: Vulnerability databases, license policies

## 🔒 Security Best Practices

### 1. Authentication

- Use strong, unique passwords
- Enable multi-factor authentication (MFA)
- Implement proper session management
- Use HTTPS for all communications
- Regularly rotate API keys and tokens

### 2. Authorization

- Follow the principle of least privilege
- Implement role-based access control (RBAC)
- Regular access reviews and audits
- Use secure token storage

### 3. Monitoring

- Monitor all authentication attempts
- Set up alerts for suspicious activity
- Regular security log reviews
- Implement automated threat response

### 4. Secrets Management

- Encrypt all secrets at rest and in transit
- Use secure key management
- Regular key rotation
- Implement access controls for secrets

### 5. Dependency Management

- Regular vulnerability scans
- Keep dependencies updated
- Monitor for security advisories
- Use dependency pinning

## 🧪 Testing

### Unit Tests

```bash
# Run security tests
pytest security/tests/ -v

# Run with coverage
pytest security/tests/ --cov=security --cov-report=html
```

### Security Tests

```bash
# Run bandit security linter
bandit -r security/

# Run safety vulnerability checker
safety check

# Run dependency vulnerability scan
python security_cli.py scan dependencies
```

## 📊 Compliance

### Supported Standards

- **SOC 2**: Security controls and monitoring
- **ISO 27001**: Information security management
- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data protection

### Audit Logging

All security events are logged with:
- Timestamp and user information
- IP address and user agent
- Action performed and result
- Threat level assessment

### Reporting

Generate compliance reports:

```python
# Security summary report
summary = security_monitor.get_security_summary()

# Audit log export
audit_log = security_monitor.get_audit_log(hours=24)

# Dependency security report
report = dependency_scanner.generate_report(scan_results)
```

## 🚨 Incident Response

### Automated Response

The security framework includes automated incident response:

1. **Authentication Breach**
   - Block affected IPs
   - Reset user sessions
   - Notify security team
   - Review access logs

2. **Data Breach**
   - Isolate affected systems
   - Preserve evidence
   - Notify legal team
   - Assess data exposure

### Manual Response

```python
# Block suspicious IP
security_monitor.block_ip("192.168.1.100", "Manual block")

# Create security alert
security_monitor.create_alert(
    AlertType.SECURITY_VIOLATION,
    ThreatLevel.HIGH,
    "Manual security alert",
    "192.168.1.100"
)

# Export evidence
security_monitor.export_events("json", "incident_evidence.json")
```

## 🔧 Integration

### API Integration

```python
# FastAPI integration
from fastapi import Depends, HTTPException
from security.auth_manager import AuthManager

auth_manager = AuthManager()

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = auth_manager.authenticate_jwt(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/protected")
def protected_route(current_user = Depends(get_current_user)):
    return {"message": "Access granted", "user": current_user.username}
```

### Webhook Integration

```python
# Security alert webhook
def security_alert_webhook(alert):
    # Send to Slack
    slack_client.chat_post_message(
        channel="#security",
        text=f"Security Alert: {alert.description}"
    )
    
    # Send to email
    send_security_email(alert)

# Register webhook
security_monitor.add_alert_callback(security_alert_webhook)
```

## 📚 API Reference

### AuthManager

```python
class AuthManager:
    def create_user(self, username: str, email: str, role: UserRole, permissions: List[str]) -> User
    def authenticate_api_key(self, api_key: str) -> Optional[User]
    def authenticate_jwt(self, token: str) -> Optional[User]
    def create_jwt_token(self, user: User) -> str
    def check_permission(self, user: User, permission: str) -> bool
```

### SecurityMonitor

```python
class SecurityMonitor:
    def record_event(self, event_type: str, source_ip: str, user_id: Optional[str] = None) -> SecurityEvent
    def create_alert(self, alert_type: AlertType, threat_level: ThreatLevel, description: str) -> SecurityAlert
    def block_ip(self, ip_address: str, reason: str)
    def get_security_summary(self) -> Dict[str, Any]
```

### SecretsManager

```python
class SecretsManager:
    def create_secret(self, name: str, value: str, secret_type: SecretType) -> str
    def get_secret_by_name(self, name: str) -> Optional[str]
    def update_secret(self, secret_id: str, value: str) -> bool
    def delete_secret(self, secret_id: str) -> bool
    def list_secrets(self, secret_type: Optional[SecretType] = None) -> List[Dict[str, Any]]
```

### DependencyScanner

```python
class DependencyScanner:
    def scan_requirements_file(self, requirements_path: str) -> Dict[str, Any]
    def scan_installed_packages(self) -> Dict[str, Any]
    def check_license_compliance(self, requirements_path: str, allowed_licenses: List[str]) -> Dict[str, Any]
    def generate_report(self, scan_results: Dict[str, Any], output_path: Optional[str] = None) -> str
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd OpenTrustEval

# Install development dependencies
pip install -r requirements_security.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest security/tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add unit tests for new features

### Security Guidelines

- Never commit secrets or sensitive data
- Use environment variables for configuration
- Follow secure coding practices
- Regular security reviews

## 📄 License

This security framework is part of OpenTrustEval and follows the same license terms.

## 🆘 Support

For security issues or questions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the security team

---

**⚠️ Security Notice**: This framework is designed for security but should be properly configured and tested in your environment. Always follow security best practices and conduct regular security audits. 