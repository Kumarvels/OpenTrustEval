"""
Dependency Scanner for OpenTrustEval
Scans dependencies for security vulnerabilities and compliance issues
"""

import os
import json
import subprocess
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Vulnerability:
    """Vulnerability information"""
    package_name: str
    version: str
    severity: str
    description: str
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    published_date: Optional[str] = None
    fixed_version: Optional[str] = None

@dataclass
class Dependency:
    """Dependency information"""
    name: str
    version: str
    license: Optional[str] = None
    source: str = "pypi"
    vulnerabilities: List[Vulnerability] = None

class DependencyScanner:
    """
    Scans Python dependencies for security vulnerabilities
    """
    
    def __init__(self):
        self.logger = logging.getLogger('DependencyScanner')
        self.vulnerability_db_url = "https://api.osv.dev/v1/query"
        self.severity_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        
    def scan_requirements_file(self, requirements_path: str) -> Dict[str, Any]:
        """Scan requirements.txt file for vulnerabilities"""
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
        
        dependencies = self._parse_requirements_file(requirements_path)
        vulnerabilities = []
        
        for dep in dependencies:
            dep_vulns = self._check_package_vulnerabilities(dep.name, dep.version)
            dep.vulnerabilities = dep_vulns
            vulnerabilities.extend(dep_vulns)
        
        return {
            'scan_timestamp': datetime.now().isoformat(),
            'requirements_file': requirements_path,
            'dependencies': [self._dependency_to_dict(dep) for dep in dependencies],
            'vulnerabilities': [self._vulnerability_to_dict(vuln) for vuln in vulnerabilities],
            'summary': self._generate_summary(dependencies, vulnerabilities)
        }
    
    def scan_installed_packages(self) -> Dict[str, Any]:
        """Scan currently installed packages for vulnerabilities"""
        dependencies = self._get_installed_packages()
        vulnerabilities = []
        
        for dep in dependencies:
            dep_vulns = self._check_package_vulnerabilities(dep.name, dep.version)
            dep.vulnerabilities = dep_vulns
            vulnerabilities.extend(dep_vulns)
        
        return {
            'scan_timestamp': datetime.now().isoformat(),
            'dependencies': [self._dependency_to_dict(dep) for dep in dependencies],
            'vulnerabilities': [self._vulnerability_to_dict(vuln) for vuln in vulnerabilities],
            'summary': self._generate_summary(dependencies, vulnerabilities)
        }
    
    def _parse_requirements_file(self, requirements_path: str) -> List[Dependency]:
        """Parse requirements.txt file"""
        dependencies = []
        
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package specification
                    if '==' in line:
                        name, version = line.split('==', 1)
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                    elif '<=' in line:
                        name, version = line.split('<=', 1)
                    elif '~=' in line:
                        name, version = line.split('~=', 1)
                    else:
                        name, version = line, "latest"
                    
                    name = name.strip()
                    version = version.strip()
                    
                    dependencies.append(Dependency(name=name, version=version))
        
        return dependencies
    
    def _get_installed_packages(self) -> List[Dependency]:
        """Get list of installed packages"""
        try:
            result = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages_data = json.loads(result.stdout)
            dependencies = []
            
            for package in packages_data:
                dependencies.append(Dependency(
                    name=package['name'],
                    version=package['version']
                ))
            
            return dependencies
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get installed packages: {e}")
            return []
    
    def _check_package_vulnerabilities(self, package_name: str, version: str) -> List[Vulnerability]:
        """Check for vulnerabilities in a specific package version"""
        vulnerabilities = []
        
        try:
            # Query OSV database
            query = {
                "package": {
                    "name": package_name,
                    "ecosystem": "PyPI"
                },
                "version": version
            }
            
            response = requests.post(self.vulnerability_db_url, json=query)
            response.raise_for_status()
            
            vulns_data = response.json().get('vulns', [])
            
            for vuln_data in vulns_data:
                vuln = self._parse_vulnerability_data(package_name, version, vuln_data)
                if vuln:
                    vulnerabilities.append(vuln)
            
        except requests.RequestException as e:
            self.logger.warning(f"Failed to check vulnerabilities for {package_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error checking vulnerabilities for {package_name}: {e}")
        
        return vulnerabilities
    
    def _parse_vulnerability_data(self, package_name: str, version: str, vuln_data: Dict) -> Optional[Vulnerability]:
        """Parse vulnerability data from OSV API"""
        try:
            # Extract basic information
            description = vuln_data.get('summary', 'No description available')
            severity = self._extract_severity(vuln_data)
            
            # Extract CVE ID
            cve_id = None
            for reference in vuln_data.get('references', []):
                if 'cve' in reference.get('url', '').lower():
                    cve_id = reference['url'].split('/')[-1]
                    break
            
            # Extract CVSS score
            cvss_score = None
            if 'database_specific' in vuln_data:
                cvss = vuln_data['database_specific'].get('cvss')
                if cvss:
                    cvss_score = cvss.get('score')
            
            # Extract fixed version
            fixed_version = None
            for affected in vuln_data.get('affected', []):
                for fixed in affected.get('fixed', []):
                    if fixed.get('ecosystem') == 'PyPI':
                        fixed_version = fixed.get('fixed')
                        break
                if fixed_version:
                    break
            
            return Vulnerability(
                package_name=package_name,
                version=version,
                severity=severity,
                description=description,
                cve_id=cve_id,
                cvss_score=cvss_score,
                published_date=vuln_data.get('published'),
                fixed_version=fixed_version
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing vulnerability data: {e}")
            return None
    
    def _extract_severity(self, vuln_data: Dict) -> str:
        """Extract severity from vulnerability data"""
        # Check database-specific severity
        if 'database_specific' in vuln_data:
            severity = vuln_data['database_specific'].get('severity', 'MEDIUM')
            if severity.upper() in self.severity_levels:
                return severity.upper()
        
        # Default to MEDIUM if no severity found
        return 'MEDIUM'
    
    def _dependency_to_dict(self, dep: Dependency) -> Dict[str, Any]:
        """Convert Dependency to dictionary"""
        return {
            'name': dep.name,
            'version': dep.version,
            'license': dep.license,
            'source': dep.source,
            'vulnerability_count': len(dep.vulnerabilities) if dep.vulnerabilities else 0
        }
    
    def _vulnerability_to_dict(self, vuln: Vulnerability) -> Dict[str, Any]:
        """Convert Vulnerability to dictionary"""
        return {
            'package_name': vuln.package_name,
            'version': vuln.version,
            'severity': vuln.severity,
            'description': vuln.description,
            'cve_id': vuln.cve_id,
            'cvss_score': vuln.cvss_score,
            'published_date': vuln.published_date,
            'fixed_version': vuln.fixed_version
        }
    
    def _generate_summary(self, dependencies: List[Dependency], vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Generate scan summary"""
        total_deps = len(dependencies)
        total_vulns = len(vulnerabilities)
        
        severity_counts = {}
        for level in self.severity_levels:
            severity_counts[level] = len([v for v in vulnerabilities if v.severity == level])
        
        critical_vulns = [v for v in vulnerabilities if v.severity == 'CRITICAL']
        high_vulns = [v for v in vulnerabilities if v.severity == 'HIGH']
        
        return {
            'total_dependencies': total_deps,
            'total_vulnerabilities': total_vulns,
            'severity_breakdown': severity_counts,
            'critical_vulnerabilities': len(critical_vulns),
            'high_vulnerabilities': len(high_vulns),
            'risk_level': self._calculate_risk_level(severity_counts)
        }
    
    def _calculate_risk_level(self, severity_counts: Dict[str, int]) -> str:
        """Calculate overall risk level"""
        if severity_counts.get('CRITICAL', 0) > 0:
            return 'CRITICAL'
        elif severity_counts.get('HIGH', 0) > 5:
            return 'HIGH'
        elif severity_counts.get('HIGH', 0) > 0 or severity_counts.get('MEDIUM', 0) > 10:
            return 'MEDIUM'
        elif severity_counts.get('MEDIUM', 0) > 0 or severity_counts.get('LOW', 0) > 20:
            return 'LOW'
        else:
            return 'SAFE'
    
    def generate_report(self, scan_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate security report"""
        report = []
        report.append("# Dependency Security Scan Report")
        report.append(f"Generated: {scan_results['scan_timestamp']}")
        report.append("")
        
        # Summary
        summary = scan_results['summary']
        report.append("## Summary")
        report.append(f"- Total Dependencies: {summary['total_dependencies']}")
        report.append(f"- Total Vulnerabilities: {summary['total_vulnerabilities']}")
        report.append(f"- Risk Level: {summary['risk_level']}")
        report.append("")
        
        # Severity breakdown
        report.append("## Severity Breakdown")
        for severity, count in summary['severity_breakdown'].items():
            if count > 0:
                report.append(f"- {severity}: {count}")
        report.append("")
        
        # Critical vulnerabilities
        critical_vulns = [v for v in scan_results['vulnerabilities'] if v['severity'] == 'CRITICAL']
        if critical_vulns:
            report.append("## Critical Vulnerabilities")
            for vuln in critical_vulns:
                report.append(f"### {vuln['package_name']} {vuln['version']}")
                report.append(f"- CVE: {vuln['cve_id'] or 'N/A'}")
                report.append(f"- Description: {vuln['description']}")
                if vuln['fixed_version']:
                    report.append(f"- Fixed in: {vuln['fixed_version']}")
                report.append("")
        
        # High vulnerabilities
        high_vulns = [v for v in scan_results['vulnerabilities'] if v['severity'] == 'HIGH']
        if high_vulns:
            report.append("## High Vulnerabilities")
            for vuln in high_vulns:
                report.append(f"### {vuln['package_name']} {vuln['version']}")
                report.append(f"- CVE: {vuln['cve_id'] or 'N/A'}")
                report.append(f"- Description: {vuln['description']}")
                if vuln['fixed_version']:
                    report.append(f"- Fixed in: {vuln['fixed_version']}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if summary['risk_level'] in ['CRITICAL', 'HIGH']:
            report.append("⚠️ **IMMEDIATE ACTION REQUIRED**")
            report.append("- Update packages with critical/high vulnerabilities")
            report.append("- Review and test updates in staging environment")
            report.append("- Consider alternative packages if updates are not available")
        elif summary['risk_level'] == 'MEDIUM':
            report.append("⚠️ **ACTION RECOMMENDED**")
            report.append("- Plan updates for packages with medium vulnerabilities")
            report.append("- Monitor for new security patches")
        else:
            report.append("✅ **No immediate action required**")
            report.append("- Continue regular security monitoring")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Security report saved to {output_path}")
        
        return report_text
    
    def check_license_compliance(self, requirements_path: str, allowed_licenses: List[str]) -> Dict[str, Any]:
        """Check license compliance"""
        dependencies = self._parse_requirements_file(requirements_path)
        compliance_issues = []
        
        for dep in dependencies:
            license_info = self._get_package_license(dep.name)
            if license_info and license_info not in allowed_licenses:
                compliance_issues.append({
                    'package': dep.name,
                    'version': dep.version,
                    'license': license_info,
                    'status': 'NON_COMPLIANT'
                })
        
        return {
            'total_packages': len(dependencies),
            'compliant_packages': len(dependencies) - len(compliance_issues),
            'non_compliant_packages': len(compliance_issues),
            'issues': compliance_issues
        }
    
    def _get_package_license(self, package_name: str) -> Optional[str]:
        """Get package license information"""
        try:
            result = subprocess.run(
                ['pip', 'show', package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith('License:'):
                    return line.split(':', 1)[1].strip()
            
            return None
            
        except subprocess.CalledProcessError:
            return None
    
    def export_results(self, scan_results: Dict[str, Any], format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export scan results in various formats"""
        if format.lower() == 'json':
            content = json.dumps(scan_results, indent=2)
        elif format.lower() == 'csv':
            content = self._export_to_csv(scan_results)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Results exported to {output_path}")
        
        return content
    
    def _export_to_csv(self, scan_results: Dict[str, Any]) -> str:
        """Export vulnerabilities to CSV format"""
        csv_lines = ['Package,Version,Severity,CVE,Description,Fixed Version']
        
        for vuln in scan_results['vulnerabilities']:
            csv_lines.append(f"{vuln['package_name']},{vuln['version']},{vuln['severity']},{vuln['cve_id'] or 'N/A'},{vuln['description'].replace(',', ';')},{vuln['fixed_version'] or 'N/A'}")
        
        return '\n'.join(csv_lines) 