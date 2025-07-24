#!/usr/bin/env python3
"""
Security CLI for OpenTrustEval
Command-line interface for security management and monitoring
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.opentrusteval.security.auth_manager import AuthManager, UserRole
from src.opentrusteval.security.oauth_provider import OAuthProvider
from src.opentrusteval.security.saml_provider import SAMLProvider
from src.opentrusteval.security.security_monitor import SecurityMonitor, ThreatLevel, AlertType
from src.opentrusteval.security.dependency_scanner import DependencyScanner
from src.opentrusteval.security.secrets_manager import SecretsManager, SecretType

def main():
    parser = argparse.ArgumentParser(description="OpenTrustEval Security Management CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Authentication management
    auth_parser = subparsers.add_parser('auth', help='Authentication management')
    auth_subparsers = auth_parser.add_subparsers(dest='auth_command', required=True)
    
    # User management
    user_parser = auth_subparsers.add_parser('user', help='User management')
    user_subparsers = user_parser.add_subparsers(dest='user_command', required=True)
    
    user_create_parser = user_subparsers.add_parser('create', help='Create user')
    user_create_parser.add_argument('--username', required=True, help='Username')
    user_create_parser.add_argument('--email', required=True, help='Email')
    user_create_parser.add_argument('--role', choices=['admin', 'user', 'readonly', 'api_user'], default='user', help='User role')
    user_create_parser.add_argument('--permissions', nargs='*', default=['read', 'write'], help='User permissions')
    
    user_list_parser = user_subparsers.add_parser('list', help='List users')
    user_delete_parser = user_subparsers.add_parser('delete', help='Delete user')
    user_delete_parser.add_argument('--user-id', required=True, help='User ID')
    
    # OAuth management
    oauth_parser = auth_subparsers.add_parser('oauth', help='OAuth management')
    oauth_subparsers = oauth_parser.add_subparsers(dest='oauth_command', required=True)
    
    oauth_list_parser = oauth_subparsers.add_parser('list', help='List OAuth providers')
    oauth_test_parser = oauth_subparsers.add_parser('test', help='Test OAuth provider')
    oauth_test_parser.add_argument('--provider', required=True, help='OAuth provider name')
    
    # SAML management
    saml_parser = auth_subparsers.add_parser('saml', help='SAML management')
    saml_subparsers = saml_parser.add_subparsers(dest='saml_command', required=True)
    
    saml_list_parser = saml_subparsers.add_parser('list', help='List SAML providers')
    saml_metadata_parser = saml_subparsers.add_parser('metadata', help='Generate SAML metadata')
    saml_metadata_parser.add_argument('--provider', required=True, help='SAML provider name')
    saml_metadata_parser.add_argument('--base-url', required=True, help='Base URL for service provider')
    
    # Security monitoring
    monitor_parser = subparsers.add_parser('monitor', help='Security monitoring')
    monitor_subparsers = monitor_parser.add_subparsers(dest='monitor_command', required=True)
    
    monitor_summary_parser = monitor_subparsers.add_parser('summary', help='Get security summary')
    monitor_alerts_parser = monitor_subparsers.add_parser('alerts', help='List recent alerts')
    monitor_alerts_parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    monitor_alerts_parser.add_argument('--severity', choices=['low', 'medium', 'high', 'critical'], help='Filter by severity')
    
    monitor_events_parser = monitor_subparsers.add_parser('events', help='List recent events')
    monitor_events_parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    
    monitor_blocks_parser = monitor_subparsers.add_parser('blocks', help='List blocked IPs')
    monitor_unblock_parser = monitor_subparsers.add_parser('unblock', help='Unblock IP')
    monitor_unblock_parser.add_argument('--ip', required=True, help='IP address to unblock')
    
    # Dependency scanning
    scan_parser = subparsers.add_parser('scan', help='Dependency scanning')
    scan_subparsers = scan_parser.add_subparsers(dest='scan_command', required=True)
    
    scan_dependencies_parser = scan_subparsers.add_parser('dependencies', help='Scan dependencies')
    scan_dependencies_parser.add_argument('--requirements', default='requirements.txt', help='Requirements file path')
    scan_dependencies_parser.add_argument('--output', help='Output file path')
    scan_dependencies_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    
    scan_installed_parser = scan_subparsers.add_parser('installed', help='Scan installed packages')
    scan_installed_parser.add_argument('--output', help='Output file path')
    scan_installed_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    
    scan_licenses_parser = scan_subparsers.add_parser('licenses', help='Check license compliance')
    scan_licenses_parser.add_argument('--requirements', default='requirements.txt', help='Requirements file path')
    scan_licenses_parser.add_argument('--allowed', nargs='*', default=['MIT', 'Apache-2.0', 'BSD-3-Clause'], help='Allowed licenses')
    
    # Secrets management
    secrets_parser = subparsers.add_parser('secrets', help='Secrets management')
    secrets_subparsers = secrets_parser.add_subparsers(dest='secrets_command', required=True)
    
    secrets_create_parser = secrets_subparsers.add_parser('create', help='Create secret')
    secrets_create_parser.add_argument('--name', required=True, help='Secret name')
    secrets_create_parser.add_argument('--value', required=True, help='Secret value')
    secrets_create_parser.add_argument('--type', choices=['api_key', 'password', 'token', 'certificate', 'database', 'oauth', 'saml'], default='api_key', help='Secret type')
    secrets_create_parser.add_argument('--description', help='Secret description')
    secrets_create_parser.add_argument('--tags', nargs='*', help='Secret tags')
    secrets_create_parser.add_argument('--expires-in-days', type=int, help='Days until expiration')
    
    secrets_list_parser = secrets_subparsers.add_parser('list', help='List secrets')
    secrets_list_parser.add_argument('--type', help='Filter by type')
    secrets_list_parser.add_argument('--tags', nargs='*', help='Filter by tags')
    secrets_list_parser.add_argument('--include-expired', action='store_true', help='Include expired secrets')
    
    secrets_get_parser = secrets_subparsers.add_parser('get', help='Get secret value')
    secrets_get_parser.add_argument('--name', required=True, help='Secret name')
    
    secrets_delete_parser = secrets_subparsers.add_parser('delete', help='Delete secret')
    secrets_delete_parser.add_argument('--name', required=True, help='Secret name')
    
    secrets_export_parser = secrets_subparsers.add_parser('export', help='Export secrets')
    secrets_export_parser.add_argument('--output', required=True, help='Output file path')
    secrets_export_parser.add_argument('--include-values', action='store_true', help='Include secret values')
    secrets_export_parser.add_argument('--format', choices=['json'], default='json', help='Output format')
    
    # Security configuration
    config_parser = subparsers.add_parser('config', help='Security configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_command', required=True)
    
    config_show_parser = config_subparsers.add_parser('show', help='Show security configuration')
    config_validate_parser = config_subparsers.add_parser('validate', help='Validate security configuration')
    config_validate_parser.add_argument('--config', default='security/security_config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'auth':
            handle_auth_command(args)
        elif args.command == 'monitor':
            handle_monitor_command(args)
        elif args.command == 'scan':
            handle_scan_command(args)
        elif args.command == 'secrets':
            handle_secrets_command(args)
        elif args.command == 'config':
            handle_config_command(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def handle_auth_command(args):
    """Handle authentication commands"""
    auth_manager = AuthManager()
    
    if args.auth_command == 'user':
        if args.user_command == 'create':
            role = UserRole(args.role)
            user = auth_manager.create_user(args.username, args.email, role, args.permissions)
            print(f"User created: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Role: {user.role.value}")
            
        elif args.user_command == 'list':
            users = auth_manager.list_users()
            print("Users:")
            for user in users:
                print(f"  {user.id}: {user.username} ({user.email}) - {user.role.value}")
                
        elif args.user_command == 'delete':
            success = auth_manager.delete_user(args.user_id)
            if success:
                print(f"User {args.user_id} deleted")
            else:
                print(f"User {args.user_id} not found")
    
    elif args.auth_command == 'oauth':
        oauth_provider = OAuthProvider()
        
        if args.oauth_command == 'list':
            providers = oauth_provider.get_supported_providers()
            print("OAuth Providers:")
            for provider in providers:
                configured = oauth_provider.is_provider_configured(provider)
                status = "✓ Configured" if configured else "✗ Not configured"
                print(f"  {provider}: {status}")
                
        elif args.oauth_command == 'test':
            if oauth_provider.is_provider_configured(args.provider):
                print(f"OAuth provider {args.provider} is properly configured")
            else:
                print(f"OAuth provider {args.provider} is not configured")
    
    elif args.auth_command == 'saml':
        saml_provider = SAMLProvider()
        
        if args.saml_command == 'list':
            providers = saml_provider.get_supported_providers()
            print("SAML Providers:")
            for provider in providers:
                configured = saml_provider.is_provider_configured(provider)
                status = "✓ Configured" if configured else "✗ Not configured"
                print(f"  {provider}: {status}")
                
        elif args.saml_command == 'metadata':
            try:
                metadata = saml_provider.create_test_metadata(args.provider, args.base_url)
                print(f"SAML Metadata for {args.provider}:")
                print(metadata)
            except Exception as e:
                print(f"Error generating metadata: {e}")

def handle_monitor_command(args):
    """Handle security monitoring commands"""
    monitor = SecurityMonitor()
    
    if args.monitor_command == 'summary':
        summary = monitor.get_security_summary()
        print("Security Summary:")
        print(f"  Total Events (24h): {summary['total_events_24h']}")
        print(f"  Active Alerts: {summary['active_alerts']}")
        print(f"  Blocked IPs: {summary['blocked_ips']}")
        print(f"  Suspicious Users: {summary['suspicious_users']}")
        print("  Threat Level Distribution:")
        for level, count in summary['threat_level_distribution'].items():
            print(f"    {level}: {count}")
    
    elif args.monitor_command == 'alerts':
        alerts = monitor.get_recent_alerts(args.hours)
        if args.severity:
            alerts = [a for a in alerts if a['threat_level'] == args.severity.upper()]
        
        print(f"Recent Alerts (last {args.hours} hours):")
        for alert in alerts:
            print(f"  {alert['alert_id']}: {alert['alert_type']} - {alert['description']}")
            print(f"    Threat Level: {alert['threat_level']}")
            print(f"    Source IP: {alert['source_ip']}")
            print(f"    Timestamp: {alert['timestamp']}")
            print()
    
    elif args.monitor_command == 'events':
        # Note: This would need to be implemented in SecurityMonitor
        print("Events listing not yet implemented")
    
    elif args.monitor_command == 'blocks':
        blocked_ips = monitor.get_blocked_ips()
        print("Blocked IPs:")
        for block in blocked_ips:
            print(f"  {block['ip_address']}: blocked until {block['blocked_until']} ({block['remaining_minutes']} minutes remaining)")
    
    elif args.monitor_command == 'unblock':
        monitor.unblock_ip(args.ip)
        print(f"IP {args.ip} unblocked")

def handle_scan_command(args):
    """Handle dependency scanning commands"""
    scanner = DependencyScanner()
    
    if args.scan_command == 'dependencies':
        results = scanner.scan_requirements_file(args.requirements)
        
        if args.output:
            scanner.export_results(results, args.format, args.output)
            print(f"Scan results exported to {args.output}")
        else:
            print("Dependency Scan Results:")
            print(f"  Total Dependencies: {results['summary']['total_dependencies']}")
            print(f"  Total Vulnerabilities: {results['summary']['total_vulnerabilities']}")
            print(f"  Risk Level: {results['summary']['risk_level']}")
            
            if results['vulnerabilities']:
                print("  Vulnerabilities:")
                for vuln in results['vulnerabilities'][:10]:  # Show first 10
                    print(f"    {vuln['package_name']} {vuln['version']}: {vuln['severity']} - {vuln['description']}")
    
    elif args.scan_command == 'installed':
        results = scanner.scan_installed_packages()
        
        if args.output:
            scanner.export_results(results, args.format, args.output)
            print(f"Scan results exported to {args.output}")
        else:
            print("Installed Packages Scan Results:")
            print(f"  Total Dependencies: {results['summary']['total_dependencies']}")
            print(f"  Total Vulnerabilities: {results['summary']['total_vulnerabilities']}")
            print(f"  Risk Level: {results['summary']['risk_level']}")
    
    elif args.scan_command == 'licenses':
        compliance = scanner.check_license_compliance(args.requirements, args.allowed)
        print("License Compliance Check:")
        print(f"  Total Packages: {compliance['total_packages']}")
        print(f"  Compliant Packages: {compliance['compliant_packages']}")
        print(f"  Non-Compliant Packages: {compliance['non_compliant_packages']}")
        
        if compliance['issues']:
            print("  Non-Compliant Packages:")
            for issue in compliance['issues']:
                print(f"    {issue['package']} {issue['version']}: {issue['license']}")

def handle_secrets_command(args):
    """Handle secrets management commands"""
    secrets_manager = SecretsManager()
    
    if args.secrets_command == 'create':
        secret_type = SecretType(args.type)
        secret_id = secrets_manager.create_secret(
            args.name, args.value, secret_type, args.description,
            args.tags, args.expires_in_days
        )
        print(f"Secret created: {secret_id}")
    
    elif args.secrets_command == 'list':
        secrets = secrets_manager.list_secrets(
            SecretType(args.type) if args.type else None,
            args.tags, args.include_expired
        )
        print("Secrets:")
        for secret in secrets:
            status = "EXPIRED" if secret['expired'] else "ACTIVE"
            print(f"  {secret['id']}: {secret['name']} ({secret['type']}) - {status}")
            if secret['description']:
                print(f"    Description: {secret['description']}")
            if secret['tags']:
                print(f"    Tags: {', '.join(secret['tags'])}")
    
    elif args.secrets_command == 'get':
        value = secrets_manager.get_secret_by_name(args.name)
        if value:
            print(f"Secret value for {args.name}: {value}")
        else:
            print(f"Secret {args.name} not found")
    
    elif args.secrets_command == 'delete':
        success = secrets_manager.delete_secret(args.name)
        if success:
            print(f"Secret {args.name} deleted")
        else:
            print(f"Secret {args.name} not found")
    
    elif args.secrets_command == 'export':
        content = secrets_manager.export_secrets(args.format, args.include_values, args.output)
        if not args.output:
            print(content)

def handle_config_command(args):
    """Handle security configuration commands"""
    if args.config_command == 'show':
        config_path = 'security/security_config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                print(f.read())
        else:
            print("Configuration file not found")
    
    elif args.config_command == 'validate':
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_sections = ['authentication', 'security_monitoring', 'secrets_management']
            for section in required_sections:
                if section not in config:
                    print(f"Warning: Missing required section '{section}'")
            
            print("Configuration validation completed")
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")

if __name__ == "__main__":
    main() 