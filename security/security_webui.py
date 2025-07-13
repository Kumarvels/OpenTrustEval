#!/usr/bin/env python3
"""
Security WebUI for OpenTrustEval
Web-based interface for security management and monitoring
"""

import os
import sys
import gradio as gr
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize security components with error handling
auth_manager = None
oauth_provider = None
saml_provider = None
security_monitor = None
dependency_scanner = None
secrets_manager = None

try:
    from security.auth_manager import AuthManager, UserRole
    auth_manager = AuthManager()
    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Security modules not available: {e}")
    SECURITY_AVAILABLE = False

if SECURITY_AVAILABLE:
    try:
        from security.oauth_provider import OAuthProvider
        oauth_provider = OAuthProvider()
    except Exception as e:
        print(f"Warning: OAuth provider not available: {e}")
    
    try:
        from security.saml_provider import SAMLProvider
        saml_provider = SAMLProvider()
    except Exception as e:
        print(f"Warning: SAML provider not available: {e}")
    
    try:
        from security.security_monitor import SecurityMonitor, ThreatLevel, AlertType
        security_monitor = SecurityMonitor()
    except Exception as e:
        print(f"Warning: Security monitor not available: {e}")
    
    try:
        from security.dependency_scanner import DependencyScanner
        dependency_scanner = DependencyScanner()
    except Exception as e:
        print(f"Warning: Dependency scanner not available: {e}")
    
    try:
        from security.secrets_manager import SecretsManager, SecretType
        secrets_manager = SecretsManager()
    except Exception as e:
        print(f"Warning: Secrets manager not available: {e}")


# High-Performance System Integration
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.independent_safety_layer import IndependentSafetyLayer
    from high_performance_system.core.human_in_the_loop_remediation import HumanInTheLoopRemediation
    
    # Initialize high-performance components
    moe_system = UltimateMoESystem()
    safety_layer = IndependentSafetyLayer()
    remediation_system = HumanInTheLoopRemediation()
    
    HIGH_PERFORMANCE_AVAILABLE = True
    print("✅ Security system integrated with high-performance components")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available: {e}")

def get_high_performance_security_status():
    """Get high-performance system security status"""
    if not HIGH_PERFORMANCE_AVAILABLE:
        return "High-performance system not available"
    
    try:
        # Get security status from high-performance system
        status = {
            'moe_system': 'active' if moe_system else 'inactive',
            'safety_layer': 'active' if safety_layer else 'inactive',
            'remediation_system': 'active' if remediation_system else 'inactive'
        }
        return json.dumps(status, indent=2)
    except Exception as e:
        return f"Error getting security status: {e}"


# --- Authentication Management Functions ---
def list_users():
    """List all users"""
    if not auth_manager:
        return "Authentication manager not available. Check dependencies and configuration."
    
    try:
        users = auth_manager.list_users()
        result = "Users:\n\n"
        for user in users:
            result += f"ID: {user.id}\n"
            result += f"Username: {user.username}\n"
            result += f"Email: {user.email}\n"
            result += f"Role: {user.role.value}\n"
            result += f"Active: {user.is_active}\n"
            result += f"Created: {user.created_at}\n"
            result += "-" * 40 + "\n"
        return result
    except Exception as e:
        return f"Error listing users: {e}"

def create_user(username, email, role, permissions):
    """Create a new user"""
    if not auth_manager:
        return "Authentication manager not available. Check dependencies and configuration."
    
    try:
        if not username or not email:
            return "Error: Username and email are required."
        
        user_role = UserRole(role)
        user = auth_manager.create_user(username, email, user_role, permissions.split(','))
        return f"User created successfully!\nID: {user.id}\nUsername: {user.username}\nEmail: {user.email}\nRole: {user.role.value}"
    except Exception as e:
        return f"Error creating user: {e}"

def delete_user(user_id):
    """Delete a user"""
    if not auth_manager:
        return "Authentication manager not available. Check dependencies and configuration."
    
    try:
        if not user_id:
            return "Error: User ID is required."
        
        success = auth_manager.delete_user(user_id)
        if success:
            return f"User {user_id} deleted successfully"
        else:
            return f"User {user_id} not found"
    except Exception as e:
        return f"Error deleting user: {e}"

def list_oauth_providers():
    """List OAuth providers"""
    if not oauth_provider:
        return "OAuth provider not available. Check dependencies and configuration."
    
    try:
        providers = oauth_provider.get_supported_providers()
        result = "OAuth Providers:\n\n"
        for provider in providers:
            configured = oauth_provider.is_provider_configured(provider)
            status = "✓ Configured" if configured else "✗ Not configured"
            result += f"{provider}: {status}\n"
        return result
    except Exception as e:
        return f"Error listing OAuth providers: {e}"

def list_saml_providers():
    """List SAML providers"""
    if not saml_provider:
        return "SAML provider not available. Check dependencies and configuration."
    
    try:
        providers = saml_provider.get_supported_providers()
        result = "SAML Providers:\n\n"
        for provider in providers:
            configured = saml_provider.is_provider_configured(provider)
            status = "✓ Configured" if configured else "✗ Not configured"
            result += f"{provider}: {status}\n"
        return result
    except Exception as e:
        return f"Error listing SAML providers: {e}"

# --- Security Monitoring Functions ---
def get_security_summary():
    """Get security summary"""
    if not security_monitor:
        return "Security monitor not available. Check dependencies and configuration."
    
    try:
        summary = security_monitor.get_security_summary()
        result = "Security Summary:\n\n"
        result += f"Total Events (24h): {summary['total_events_24h']}\n"
        result += f"Active Alerts: {summary['active_alerts']}\n"
        result += f"Blocked IPs: {summary['blocked_ips']}\n"
        result += f"Suspicious Users: {summary['suspicious_users']}\n\n"
        result += "Threat Level Distribution:\n"
        for level, count in summary['threat_level_distribution'].items():
            result += f"  {level}: {count}\n"
        return result
    except Exception as e:
        return f"Error getting security summary: {e}"

def get_recent_alerts(hours, severity):
    """Get recent alerts"""
    if not security_monitor:
        return "Security monitor not available. Check dependencies and configuration."
    
    try:
        alerts = security_monitor.get_recent_alerts(int(hours))
        if severity and severity != "all":
            alerts = [a for a in alerts if a['threat_level'] == severity.upper()]
        
        result = f"Recent Alerts (last {hours} hours):\n\n"
        for alert in alerts:
            result += f"Alert ID: {alert['alert_id']}\n"
            result += f"Type: {alert['alert_type']}\n"
            result += f"Description: {alert['description']}\n"
            result += f"Threat Level: {alert['threat_level']}\n"
            result += f"Source IP: {alert['source_ip']}\n"
            result += f"Timestamp: {alert['timestamp']}\n"
            result += "-" * 40 + "\n"
        return result
    except Exception as e:
        return f"Error getting alerts: {e}"

def get_blocked_ips():
    """Get blocked IPs"""
    if not security_monitor:
        return "Security monitor not available. Check dependencies and configuration."
    
    try:
        blocked_ips = security_monitor.get_blocked_ips()
        result = "Blocked IPs:\n\n"
        for block in blocked_ips:
            result += f"IP: {block['ip_address']}\n"
            result += f"Blocked until: {block['blocked_until']}\n"
            result += f"Remaining minutes: {block['remaining_minutes']}\n"
            result += "-" * 40 + "\n"
        return result
    except Exception as e:
        return f"Error getting blocked IPs: {e}"

def unblock_ip(ip_address):
    """Unblock an IP"""
    if not security_monitor:
        return "Security monitor not available. Check dependencies and configuration."
    
    try:
        if not ip_address:
            return "Error: IP address is required."
        
        security_monitor.unblock_ip(ip_address)
        return f"IP {ip_address} unblocked successfully"
    except Exception as e:
        return f"Error unblocking IP: {e}"

# --- Dependency Scanning Functions ---
def scan_dependencies(requirements_path, output_format):
    """Scan dependencies"""
    if not dependency_scanner:
        return "Dependency scanner not available. Check dependencies and configuration."
    
    try:
        if not os.path.exists(requirements_path):
            return f"Error: Requirements file not found: {requirements_path}"
        
        results = dependency_scanner.scan_requirements_file(requirements_path)
        
        result = "Dependency Scan Results:\n\n"
        result += f"Total Dependencies: {results['summary']['total_dependencies']}\n"
        result += f"Total Vulnerabilities: {results['summary']['total_vulnerabilities']}\n"
        result += f"Risk Level: {results['summary']['risk_level']}\n\n"
        
        if results['vulnerabilities']:
            result += "Vulnerabilities:\n"
            for vuln in results['vulnerabilities'][:20]:  # Show first 20
                result += f"  {vuln['package_name']} {vuln['version']}: {vuln['severity']}\n"
                result += f"    Description: {vuln['description']}\n"
                if vuln['cve_id']:
                    result += f"    CVE: {vuln['cve_id']}\n"
                if vuln['fixed_version']:
                    result += f"    Fixed in: {vuln['fixed_version']}\n"
                result += "\n"
        
        return result
    except Exception as e:
        return f"Error scanning dependencies: {e}"

def scan_installed_packages():
    """Scan installed packages"""
    if not dependency_scanner:
        return "Dependency scanner not available. Check dependencies and configuration."
    
    try:
        results = dependency_scanner.scan_installed_packages()
        
        result = "Installed Packages Scan Results:\n\n"
        result += f"Total Dependencies: {results['summary']['total_dependencies']}\n"
        result += f"Total Vulnerabilities: {results['summary']['total_vulnerabilities']}\n"
        result += f"Risk Level: {results['summary']['risk_level']}\n\n"
        
        if results['vulnerabilities']:
            result += "Vulnerabilities:\n"
            for vuln in results['vulnerabilities'][:20]:  # Show first 20
                result += f"  {vuln['package_name']} {vuln['version']}: {vuln['severity']}\n"
                result += f"    Description: {vuln['description']}\n"
                if vuln['cve_id']:
                    result += f"    CVE: {vuln['cve_id']}\n"
                if vuln['fixed_version']:
                    result += f"    Fixed in: {vuln['fixed_version']}\n"
                result += "\n"
        
        return result
    except Exception as e:
        return f"Error scanning installed packages: {e}"

def check_license_compliance(requirements_path, allowed_licenses):
    """Check license compliance"""
    if not dependency_scanner:
        return "Dependency scanner not available. Check dependencies and configuration."
    
    try:
        if not os.path.exists(requirements_path):
            return f"Error: Requirements file not found: {requirements_path}"
        
        allowed_list = [license.strip() for license in allowed_licenses.split(',')]
        compliance = dependency_scanner.check_license_compliance(requirements_path, allowed_list)
        
        result = "License Compliance Check:\n\n"
        result += f"Total Packages: {compliance['total_packages']}\n"
        result += f"Compliant Packages: {compliance['compliant_packages']}\n"
        result += f"Non-Compliant Packages: {compliance['non_compliant_packages']}\n\n"
        
        if compliance['issues']:
            result += "Non-Compliant Packages:\n"
            for issue in compliance['issues']:
                result += f"  {issue['package']} {issue['version']}: {issue['license']}\n"
        
        return result
    except Exception as e:
        return f"Error checking license compliance: {e}"

# --- Secrets Management Functions ---
def list_secrets(secret_type, include_expired):
    """List secrets"""
    if not secrets_manager:
        return "Secrets manager not available. Check dependencies and configuration."
    
    try:
        secrets = secrets_manager.list_secrets(
            SecretType(secret_type) if secret_type and secret_type != "all" else None,
            None, include_expired
        )
        
        result = "Secrets:\n\n"
        for secret in secrets:
            status = "EXPIRED" if secret['expired'] else "ACTIVE"
            result += f"ID: {secret['id']}\n"
            result += f"Name: {secret['name']}\n"
            result += f"Type: {secret['type']}\n"
            result += f"Status: {status}\n"
            if secret['description']:
                result += f"Description: {secret['description']}\n"
            if secret['tags']:
                result += f"Tags: {', '.join(secret['tags'])}\n"
            result += "-" * 40 + "\n"
        
        return result
    except Exception as e:
        return f"Error listing secrets: {e}"

def create_secret(name, value, secret_type, description, tags, expires_in_days):
    """Create a secret"""
    if not secrets_manager:
        return "Secrets manager not available. Check dependencies and configuration."
    
    try:
        if not name or not value:
            return "Error: Secret name and value are required."
        
        secret_type_enum = SecretType(secret_type)
        tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
        expires_days = int(expires_in_days) if expires_in_days else None
        
        secret_id = secrets_manager.create_secret(
            name, value, secret_type_enum, description, tags_list, expires_days
        )
        return f"Secret created successfully!\nID: {secret_id}\nName: {name}\nType: {secret_type}"
    except Exception as e:
        return f"Error creating secret: {e}"

def get_secret_value(name):
    """Get secret value"""
    if not secrets_manager:
        return "Secrets manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Secret name is required."
        
        value = secrets_manager.get_secret_by_name(name)
        if value:
            return f"Secret value for {name}: {value}"
        else:
            return f"Secret {name} not found"
    except Exception as e:
        return f"Error getting secret: {e}"

def delete_secret(name):
    """Delete a secret"""
    if not secrets_manager:
        return "Secrets manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Secret name is required."
        
        success = secrets_manager.delete_secret(name)
        if success:
            return f"Secret {name} deleted successfully"
        else:
            return f"Secret {name} not found"
    except Exception as e:
        return f"Error deleting secret: {e}"

def get_secrets_statistics():
    """Get secrets statistics"""
    if not secrets_manager:
        return "Secrets manager not available. Check dependencies and configuration."
    
    try:
        stats = secrets_manager.get_statistics()
        
        result = "Secrets Statistics:\n\n"
        result += f"Total Secrets: {stats['total_secrets']}\n"
        result += f"Expired Secrets: {stats['expired_secrets']}\n"
        result += f"Expiring Soon (7 days): {stats['expiring_soon']}\n"
        result += f"Storage Size: {stats['storage_size_mb']:.2f} MB\n\n"
        result += "Type Distribution:\n"
        for secret_type, count in stats['type_distribution'].items():
            result += f"  {secret_type}: {count}\n"
        
        return result
    except Exception as e:
        return f"Error getting statistics: {e}"

# --- Configuration Functions ---
def show_security_config():
    """Show security configuration"""
    try:
        config_path = 'security/security_config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return f.read()
        else:
            return "Configuration file not found. Create security/security_config.yaml"
    except Exception as e:
        return f"Error reading configuration: {e}"

def validate_security_config():
    """Validate security configuration"""
    try:
        config_path = 'security/security_config.yaml'
        if not os.path.exists(config_path):
            return "Configuration file not found. Create security/security_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_sections = ['authentication', 'security_monitoring', 'secrets_management']
        validation_results = []
        
        for section in required_sections:
            if section in config:
                validation_results.append(f"✓ {section}: Present")
            else:
                validation_results.append(f"✗ {section}: Missing")
        
        return "Configuration Validation Results:\n\n" + "\n".join(validation_results)
    except Exception as e:
        return f"Configuration validation failed: {e}"

# --- Gradio Interface ---
with gr.Blocks(title="OpenTrustEval Security Management") as demo:
    gr.Markdown("# 🔒 OpenTrustEval Security Management")
    
    if not SECURITY_AVAILABLE:
        gr.Markdown("⚠️ **Security modules not available.** Please install required dependencies.")
        gr.Markdown("Required: `security` module with proper configuration")
    
    with gr.Tab("Authentication"):
        gr.Markdown("## User Management")
        with gr.Row():
            with gr.Column():
                list_users_btn = gr.Button("List Users")
                list_users_output = gr.Textbox(label="Users", lines=15)
                list_users_btn.click(list_users, outputs=list_users_output)
            
            with gr.Column():
                gr.Markdown("### Create User")
                username_input = gr.Textbox(label="Username")
                email_input = gr.Textbox(label="Email")
                role_input = gr.Dropdown(choices=["admin", "user", "readonly", "api_user"], label="Role", value="user")
                permissions_input = gr.Textbox(label="Permissions (comma-separated)", value="read,write")
                create_user_btn = gr.Button("Create User")
                create_user_output = gr.Textbox(label="Result", lines=5)
                create_user_btn.click(
                    create_user,
                    inputs=[username_input, email_input, role_input, permissions_input],
                    outputs=create_user_output
                )
        
        gr.Markdown("### Delete User")
        user_id_input = gr.Textbox(label="User ID")
        delete_user_btn = gr.Button("Delete User")
        delete_user_output = gr.Textbox(label="Result", lines=3)
        delete_user_btn.click(delete_user, inputs=user_id_input, outputs=delete_user_output)
        
        gr.Markdown("## OAuth Providers")
        oauth_btn = gr.Button("List OAuth Providers")
        oauth_output = gr.Textbox(label="OAuth Providers", lines=10)
        oauth_btn.click(list_oauth_providers, outputs=oauth_output)
        
        gr.Markdown("## SAML Providers")
        saml_btn = gr.Button("List SAML Providers")
        saml_output = gr.Textbox(label="SAML Providers", lines=10)
        saml_btn.click(list_saml_providers, outputs=saml_output)
    
    with gr.Tab("Security Monitoring"):
        gr.Markdown("## Security Overview")
        summary_btn = gr.Button("Get Security Summary")
        summary_output = gr.Textbox(label="Security Summary", lines=15)
        summary_btn.click(get_security_summary, outputs=summary_output)
        
        gr.Markdown("## Recent Alerts")
        with gr.Row():
            hours_input = gr.Slider(minimum=1, maximum=168, value=24, step=1, label="Hours to look back")
            severity_input = gr.Dropdown(choices=["all", "low", "medium", "high", "critical"], label="Severity", value="all")
        alerts_btn = gr.Button("Get Recent Alerts")
        alerts_output = gr.Textbox(label="Recent Alerts", lines=15)
        alerts_btn.click(get_recent_alerts, inputs=[hours_input, severity_input], outputs=alerts_output)
        
        gr.Markdown("## IP Management")
        with gr.Row():
            with gr.Column():
                blocked_ips_btn = gr.Button("List Blocked IPs")
                blocked_ips_output = gr.Textbox(label="Blocked IPs", lines=10)
                blocked_ips_btn.click(get_blocked_ips, outputs=blocked_ips_output)
            
            with gr.Column():
                gr.Markdown("### Unblock IP")
                ip_input = gr.Textbox(label="IP Address")
                unblock_btn = gr.Button("Unblock IP")
                unblock_output = gr.Textbox(label="Result", lines=3)
                unblock_btn.click(unblock_ip, inputs=ip_input, outputs=unblock_output)
    
    with gr.Tab("Dependency Scanning"):
        gr.Markdown("## Scan Dependencies")
        with gr.Row():
            requirements_path = gr.Textbox(label="Requirements File Path", value="requirements.txt")
            output_format = gr.Dropdown(choices=["json", "csv"], label="Output Format", value="json")
        scan_deps_btn = gr.Button("Scan Dependencies")
        scan_deps_output = gr.Textbox(label="Scan Results", lines=20)
        scan_deps_btn.click(scan_dependencies, inputs=[requirements_path, output_format], outputs=scan_deps_output)
        
        gr.Markdown("## Scan Installed Packages")
        scan_installed_btn = gr.Button("Scan Installed Packages")
        scan_installed_output = gr.Textbox(label="Scan Results", lines=20)
        scan_installed_btn.click(scan_installed_packages, outputs=scan_installed_output)
        
        gr.Markdown("## License Compliance")
        with gr.Row():
            license_requirements = gr.Textbox(label="Requirements File Path", value="requirements.txt")
            allowed_licenses = gr.Textbox(label="Allowed Licenses (comma-separated)", value="MIT,Apache-2.0,BSD-3-Clause")
        license_btn = gr.Button("Check License Compliance")
        license_output = gr.Textbox(label="Compliance Results", lines=15)
        license_btn.click(check_license_compliance, inputs=[license_requirements, allowed_licenses], outputs=license_output)
    
    with gr.Tab("Secrets Management"):
        gr.Markdown("## Secrets Overview")
        secrets_stats_btn = gr.Button("Get Statistics")
        secrets_stats_output = gr.Textbox(label="Statistics", lines=10)
        secrets_stats_btn.click(get_secrets_statistics, outputs=secrets_stats_output)
        
        gr.Markdown("## List Secrets")
        with gr.Row():
            secret_type_filter = gr.Dropdown(choices=["all", "api_key", "password", "token", "certificate", "database", "oauth", "saml"], label="Type Filter", value="all")
            include_expired = gr.Checkbox(label="Include Expired", value=False)
        list_secrets_btn = gr.Button("List Secrets")
        list_secrets_output = gr.Textbox(label="Secrets", lines=15)
        list_secrets_btn.click(list_secrets, inputs=[secret_type_filter, include_expired], outputs=list_secrets_output)
        
        gr.Markdown("## Create Secret")
        with gr.Row():
            with gr.Column():
                secret_name = gr.Textbox(label="Secret Name")
                secret_value = gr.Textbox(label="Secret Value", type="password")
                secret_type = gr.Dropdown(choices=["api_key", "password", "token", "certificate", "database", "oauth", "saml"], label="Type", value="api_key")
                secret_description = gr.Textbox(label="Description")
                secret_tags = gr.Textbox(label="Tags (comma-separated)")
                secret_expires = gr.Textbox(label="Expires in Days (optional)")
                create_secret_btn = gr.Button("Create Secret")
                create_secret_output = gr.Textbox(label="Result", lines=5)
                create_secret_btn.click(
                    create_secret,
                    inputs=[secret_name, secret_value, secret_type, secret_description, secret_tags, secret_expires],
                    outputs=create_secret_output
                )
        
        gr.Markdown("## Manage Secrets")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Get Secret Value")
                get_secret_name = gr.Textbox(label="Secret Name")
                get_secret_btn = gr.Button("Get Value")
                get_secret_output = gr.Textbox(label="Secret Value", lines=3)
                get_secret_btn.click(get_secret_value, inputs=get_secret_name, outputs=get_secret_output)
            
            with gr.Column():
                gr.Markdown("### Delete Secret")
                delete_secret_name = gr.Textbox(label="Secret Name")
                delete_secret_btn = gr.Button("Delete Secret")
                delete_secret_output = gr.Textbox(label="Result", lines=3)
                delete_secret_btn.click(delete_secret, inputs=delete_secret_name, outputs=delete_secret_output)
    
    with gr.Tab("Configuration"):
        gr.Markdown("## Security Configuration")
        config_btn = gr.Button("Show Configuration")
        config_output = gr.Textbox(label="Configuration", lines=20)
        config_btn.click(show_security_config, outputs=config_output)
        
        gr.Markdown("## Validate Configuration")
        validate_btn = gr.Button("Validate Configuration")
        validate_output = gr.Textbox(label="Validation Results", lines=10)
        validate_btn.click(validate_security_config, outputs=validate_output)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenTrustEval Security WebUI")
    parser.add_argument("--server_port", type=int, default=7863, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    args = parser.parse_args()
    
    print(f"Starting Security WebUI on http://{args.server_name}:{args.server_port}")
    demo.launch(server_name=args.server_name, server_port=args.server_port)

if __name__ == "__main__":
    main() 