#!/usr/bin/env python3
"""
Workflow Problem Resolver for OpenTrustEval
Step-by-step problem resolution guidance based on diagnostic results
"""

import json
import os
import sys
import subprocess
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio

class WorkflowProblemResolver:
    """Step-by-step problem resolver for OpenTrustEval issues"""
    
    def __init__(self):
        self.solutions = self.load_solutions()
        self.resolution_steps = []
    
    def load_solutions(self) -> Dict[str, Any]:
        """Load problem solutions database"""
        return {
            "System Environment": {
                "python_version": {
                    "problem": "Python version too old",
                    "solution": "Upgrade to Python 3.8+",
                    "commands": [
                        "python --version",
                        "# Install Python 3.8+ from python.org or use pyenv"
                    ]
                },
                "missing_packages": {
                    "problem": "Missing required packages",
                    "solution": "Install missing dependencies",
                    "commands": [
                        "pip install fastapi uvicorn streamlit pandas numpy plotly requests",
                        "pip install -r requirements.txt"
                    ]
                },
                "low_memory": {
                    "problem": "Insufficient system memory",
                    "solution": "Free up memory or increase system resources",
                    "commands": [
                        "tasklist | findstr python",
                        "# Close unnecessary applications",
                        "# Restart system if needed"
                    ]
                }
            },
            "Data Uploads": {
                "permissions": {
                    "problem": "Upload directory permissions issue",
                    "solution": "Fix directory permissions",
                    "commands": [
                        "mkdir uploads",
                        "chmod 755 uploads",
                        "# On Windows: Right-click uploads folder -> Properties -> Security"
                    ]
                },
                "dataset_connector": {
                    "problem": "Dataset connector not available",
                    "solution": "Install or fix dataset integration",
                    "commands": [
                        "pip install pandas numpy",
                        "# Check data_engineering/dataset_integration.py"
                    ]
                }
            },
            "Data Engineering": {
                "trust_scoring": {
                    "problem": "Trust scoring engine failure",
                    "solution": "Fix trust scoring components",
                    "commands": [
                        "pip install scipy scikit-learn",
                        "python -c \"from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine; print('Trust scoring OK')\""
                    ]
                },
                "database": {
                    "problem": "Database connectivity issue",
                    "solution": "Fix database connection",
                    "commands": [
                        "pip install sqlite3",
                        "# Check database file permissions",
                        "# Verify database path"
                    ]
                }
            },
            "LLM Engineering": {
                "llm_lifecycle": {
                    "problem": "LLM lifecycle manager failure",
                    "solution": "Fix LLM lifecycle components",
                    "commands": [
                        "pip install pyyaml",
                        "python -c \"from llm_engineering.llm_lifecycle import LLMLifecycleManager; print('LLM lifecycle OK')\""
                    ]
                },
                "config": {
                    "problem": "LLM configuration missing",
                    "solution": "Create LLM configuration",
                    "commands": [
                        "mkdir -p llm_engineering/configs",
                        "# Create llm_engineering/configs/llm_providers.yaml"
                    ]
                }
            },
            "High Performance System": {
                "moe_system": {
                    "problem": "MoE system initialization failure",
                    "solution": "Fix MoE system components",
                    "commands": [
                        "pip install numpy asyncio",
                        "python -c \"from high_performance_system.core.ultimate_moe_system import UltimateMoESystem; print('MoE system OK')\""
                    ]
                },
                "expert_ensemble": {
                    "problem": "Expert ensemble failure",
                    "solution": "Fix expert ensemble components",
                    "commands": [
                        "python -c \"from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble; print('Expert ensemble OK')\""
                    ]
                }
            },
            "Security System": {
                "auth_manager": {
                    "problem": "Authentication manager failure",
                    "solution": "Fix authentication components",
                    "commands": [
                        "pip install cryptography",
                        "python -c \"from security.auth_manager import AuthManager; print('Auth manager OK')\""
                    ]
                },
                "security_webui": {
                    "problem": "Security web UI failure",
                    "solution": "Fix security web UI",
                    "commands": [
                        "pip install gradio",
                        "python -c \"from security.security_webui import get_high_performance_security_status; print('Security web UI OK')\""
                    ]
                }
            },
            "MCP Server": {
                "server_not_running": {
                    "problem": "MCP server not running",
                    "solution": "Start MCP server",
                    "commands": [
                        "python mcp_server/server.py",
                        "# Or check if server is running on different port"
                    ]
                },
                "connectivity": {
                    "problem": "MCP server connectivity issue",
                    "solution": "Check server configuration",
                    "commands": [
                        "netstat -an | findstr 8000",
                        "curl http://localhost:8000/health",
                        "# Check firewall settings"
                    ]
                }
            },
            "Production Server": {
                "server_not_running": {
                    "problem": "Production server not running",
                    "solution": "Start production server",
                    "commands": [
                        "python superfast_production_server.py",
                        "# Check if server is running on port 8003"
                    ]
                },
                "performance_endpoint": {
                    "problem": "Performance endpoint failure",
                    "solution": "Check server endpoints",
                    "commands": [
                        "curl http://localhost:8003/health",
                        "curl http://localhost:8003/performance",
                        "# Check server logs for errors"
                    ]
                }
            },
            "Tests": {
                "test_failure": {
                    "problem": "Test suite failure",
                    "solution": "Fix failing tests",
                    "commands": [
                        "pip install pytest",
                        "python -m pytest simple_unit_test.py -v",
                        "# Check test dependencies"
                    ]
                },
                "missing_tests": {
                    "problem": "Test files missing",
                    "solution": "Create missing test files",
                    "commands": [
                        "mkdir -p tests",
                        "# Create basic test files"
                    ]
                }
            },
            "Plugins": {
                "plugin_loader": {
                    "problem": "Plugin loader failure",
                    "solution": "Fix plugin loading",
                    "commands": [
                        "python -c \"from plugins.plugin_loader import load_plugins; print('Plugin loader OK')\""
                    ]
                },
                "missing_plugins": {
                    "problem": "Plugin files missing",
                    "solution": "Create plugin files",
                    "commands": [
                        "mkdir -p plugins",
                        "# Create example_plugin.py"
                    ]
                }
            },
            "Analytics & Dashboards": {
                "dashboard_import": {
                    "problem": "Dashboard import failure",
                    "solution": "Fix dashboard components",
                    "commands": [
                        "pip install streamlit plotly",
                        "python -c \"from high_performance_system.analytics.ultimate_analytics_dashboard import UltimateAnalyticsDashboard; print('Dashboard OK')\""
                    ]
                },
                "missing_dashboards": {
                    "problem": "Dashboard files missing",
                    "solution": "Create dashboard files",
                    "commands": [
                        "mkdir -p high_performance_system/analytics",
                        "# Create dashboard files"
                    ]
                }
            }
        }
    
    def analyze_diagnostic_report(self, report_file: str) -> Dict[str, Any]:
        """Analyze diagnostic report and generate resolution plan"""
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
        except FileNotFoundError:
            print(f"❌ Diagnostic report not found: {report_file}")
            return {}
        
        analysis = {
            "summary": report.get("summary", {}),
            "issues": [],
            "resolution_plan": [],
            "priority_actions": []
        }
        
        # Analyze each result
        for result in report.get("results", []):
            component = result["component"]
            status = result["status"]
            message = result["message"]
            details = result.get("details", {})
            
            if status == "FAIL":
                issue = self.identify_issue(component, details)
                if issue:
                    analysis["issues"].append({
                        "component": component,
                        "issue": issue,
                        "message": message,
                        "details": details
                    })
                    
                    resolution = self.get_resolution(component, issue)
                    if resolution:
                        analysis["resolution_plan"].append({
                            "component": component,
                            "issue": issue,
                            "resolution": resolution,
                            "priority": "HIGH"
                        })
                        analysis["priority_actions"].append(resolution)
            
            elif status == "WARNING":
                issue = self.identify_issue(component, details)
                if issue:
                    resolution = self.get_resolution(component, issue)
                    if resolution:
                        analysis["resolution_plan"].append({
                            "component": component,
                            "issue": issue,
                            "resolution": resolution,
                            "priority": "MEDIUM"
                        })
        
        return analysis
    
    def identify_issue(self, component: str, details: Dict[str, Any]) -> Optional[str]:
        """Identify specific issue from component details"""
        if component not in self.solutions:
            return None
        
        component_solutions = self.solutions[component]
        
        # Check for specific issues based on details
        if component == "System Environment":
            if "python_version" in details and not details.get("python_version", "").startswith("3.8"):
                return "python_version"
            if any("_available" in k and not v for k, v in details.items()):
                return "missing_packages"
            if "memory_available" in details:
                try:
                    memory_gb = float(details["memory_available"].split()[0])
                    if memory_gb < 2:
                        return "low_memory"
                except:
                    pass
        
        elif component == "Data Uploads":
            if not details.get("write_permissions", False):
                return "permissions"
            if not details.get("dataset_connector_available", False):
                return "dataset_connector"
        
        elif component == "Data Engineering":
            if details.get("trust_scoring_test") == "FAIL":
                return "trust_scoring"
            if not details.get("database_connectivity", False):
                return "database"
        
        elif component == "LLM Engineering":
            if details.get("llm_lifecycle_manager") == "FAIL":
                return "llm_lifecycle"
            if not details.get("llm_config_exists", False):
                return "config"
        
        elif component == "High Performance System":
            if details.get("moe_system_test") == "FAIL":
                return "moe_system"
            if details.get("expert_ensemble") == "FAIL":
                return "expert_ensemble"
        
        elif component == "Security System":
            if details.get("security_webui") == "FAIL":
                return "security_webui"
        
        elif component == "MCP Server":
            if not details.get("mcp_server_running", False):
                return "server_not_running"
            if not details.get("mcp_server_health", False):
                return "connectivity"
        
        elif component == "Production Server":
            if not details.get("production_server_running", False):
                return "server_not_running"
            if details.get("performance_endpoint") == "FAIL":
                return "performance_endpoint"
        
        elif component == "Tests":
            if details.get("simple_test_run") == "FAIL":
                return "test_failure"
            if not any(details.get(f"{f}_exists", False) for f in ["test_high_performance_system.py", "simple_unit_test.py"]):
                return "missing_tests"
        
        elif component == "Plugins":
            if details.get("plugin_loader") == "FAIL":
                return "plugin_loader"
            if details.get("plugins_loaded", 0) == 0:
                return "missing_plugins"
        
        elif component == "Analytics & Dashboards":
            if details.get("ultimate_dashboard") == "FAIL":
                return "dashboard_import"
            if not any(details.get(f"{f}_exists", False) for f in ["high_performance_system/analytics/ultimate_analytics_dashboard.py"]):
                return "missing_dashboards"
        
        return None
    
    def get_resolution(self, component: str, issue: str) -> Optional[Dict[str, Any]]:
        """Get resolution for specific issue"""
        if component in self.solutions and issue in self.solutions[component]:
            return self.solutions[component][issue]
        return None
    
    def print_resolution_plan(self, analysis: Dict[str, Any]):
        """Print step-by-step resolution plan"""
        print("\n" + "=" * 80)
        print("🔧 OPENTRUSTEVAL PROBLEM RESOLUTION PLAN")
        print("=" * 80)
        
        summary = analysis.get("summary", {})
        print(f"\n📊 DIAGNOSTIC SUMMARY:")
        print(f"   Total Issues: {len(analysis.get('issues', []))}")
        print(f"   High Priority: {len([r for r in analysis.get('resolution_plan', []) if r['priority'] == 'HIGH'])}")
        print(f"   Medium Priority: {len([r for r in analysis.get('resolution_plan', []) if r['priority'] == 'MEDIUM'])}")
        
        if not analysis.get("issues"):
            print("\n✅ No issues found! System is working correctly.")
            return
        
        print(f"\n🚨 ISSUES FOUND:")
        for issue in analysis.get("issues", []):
            print(f"   ❌ {issue['component']}: {issue['issue']}")
            print(f"      Message: {issue['message']}")
        
        print(f"\n🎯 RESOLUTION PLAN:")
        for i, resolution in enumerate(analysis.get("resolution_plan", []), 1):
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            emoji = priority_emoji.get(resolution["priority"], "❓")
            
            print(f"\n   {i}. {emoji} {resolution['priority']} PRIORITY")
            print(f"      Component: {resolution['component']}")
            print(f"      Issue: {resolution['issue']}")
            print(f"      Problem: {resolution['resolution']['problem']}")
            print(f"      Solution: {resolution['resolution']['solution']}")
            
            if 'commands' in resolution['resolution']:
                print(f"      Commands:")
                for cmd in resolution['resolution']['commands']:
                    if cmd.startswith('#'):
                        print(f"         {cmd}")
                    else:
                        print(f"         $ {cmd}")
        
        print(f"\n⚡ QUICK ACTIONS:")
        for i, action in enumerate(analysis.get("priority_actions", [])[:5], 1):
            print(f"   {i}. {action['solution']}")
            if 'commands' in action:
                print(f"      $ {action['commands'][0]}")
    
    def execute_resolution(self, component: str, issue: str, interactive: bool = True) -> bool:
        """Execute resolution for a specific issue"""
        resolution = self.get_resolution(component, issue)
        if not resolution:
            print(f"❌ No resolution found for {component}:{issue}")
            return False
        
        print(f"\n🔧 Executing resolution for {component}:{issue}")
        print(f"Problem: {resolution['problem']}")
        print(f"Solution: {resolution['solution']}")
        
        if not resolution.get('commands'):
            print("✅ No commands to execute")
            return True
        
        if interactive:
            response = input("\nExecute these commands? (y/n): ").lower()
            if response != 'y':
                print("⏭️ Skipping command execution")
                return False
        
        success = True
        for cmd in resolution['commands']:
            if cmd.startswith('#'):
                print(f"   {cmd}")
                continue
            
            print(f"   Executing: {cmd}")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"   ✅ Success: {result.stdout.strip()}")
                else:
                    print(f"   ❌ Failed: {result.stderr.strip()}")
                    success = False
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Timeout: Command took too long")
                success = False
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                success = False
        
        return success
    
    def run_interactive_resolution(self, analysis: Dict[str, Any]):
        """Run interactive resolution process"""
        if not analysis.get("issues"):
            print("✅ No issues to resolve!")
            return
        
        print(f"\n🎯 INTERACTIVE RESOLUTION")
        print("=" * 40)
        
        resolution_plan = analysis.get("resolution_plan", [])
        
        for i, resolution in enumerate(resolution_plan, 1):
            print(f"\n{i}/{len(resolution_plan)}: {resolution['component']} - {resolution['issue']}")
            print(f"Priority: {resolution['priority']}")
            print(f"Solution: {resolution['resolution']['solution']}")
            
            response = input("Execute this resolution? (y/n/s=skip all): ").lower()
            
            if response == 's':
                print("⏭️ Skipping all remaining resolutions")
                break
            elif response == 'y':
                success = self.execute_resolution(resolution['component'], resolution['issue'], interactive=False)
                if success:
                    print(f"✅ Resolution {i} completed successfully")
                else:
                    print(f"❌ Resolution {i} failed")
                    retry = input("Retry this resolution? (y/n): ").lower()
                    if retry == 'y':
                        self.execute_resolution(resolution['component'], resolution['issue'], interactive=False)
            else:
                print("⏭️ Skipping this resolution")
        
        print(f"\n🎉 Resolution process completed!")
        print("Run the diagnostic again to verify fixes.")

def main():
    """Main resolution execution"""
    resolver = WorkflowProblemResolver()
    
    # Check for diagnostic report
    report_files = [f for f in os.listdir('.') if f.startswith('workflow_diagnostic_report_') and f.endswith('.json')]
    
    if not report_files:
        print("❌ No diagnostic report found!")
        print("Please run the diagnostic first:")
        print("   python complete_workflow_diagnostic.py")
        return
    
    # Use the most recent report
    latest_report = max(report_files)
    print(f"📋 Using diagnostic report: {latest_report}")
    
    # Analyze the report
    analysis = resolver.analyze_diagnostic_report(latest_report)
    
    # Print resolution plan
    resolver.print_resolution_plan(analysis)
    
    # Offer interactive resolution
    if analysis.get("issues"):
        response = input("\nRun interactive resolution? (y/n): ").lower()
        if response == 'y':
            resolver.run_interactive_resolution(analysis)
        else:
            print("\n💡 To run resolution later, use:")
            print("   python workflow_problem_resolver.py")

if __name__ == "__main__":
    main() 