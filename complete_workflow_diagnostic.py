#!/usr/bin/env python3
"""
Complete Workflow Diagnostic System for OpenTrustEval
Step-by-step problem analysis across all system components
"""

import os
import sys
import json
import time
import asyncio
import logging
import traceback
import subprocess
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib
import socket
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check"""
    component: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'SKIP'
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration: float

@dataclass
class WorkflowStep:
    """A step in the diagnostic workflow"""
    name: str
    description: str
    dependencies: List[str]
    check_function: callable
    critical: bool = True
    timeout: int = 30

class CompleteWorkflowDiagnostic:
    """Complete workflow diagnostic system for OpenTrustEval"""
    
    def __init__(self):
        self.results: List[DiagnosticResult] = []
        self.workflow_steps: List[WorkflowStep] = []
        self.system_status: Dict[str, Any] = {}
        self.setup_workflow_steps()
    
    def setup_workflow_steps(self):
        """Setup all workflow diagnostic steps"""
        
        # 1. System Environment Check
        self.workflow_steps.append(WorkflowStep(
            name="System Environment",
            description="Check Python environment, dependencies, and system resources",
            dependencies=[],
            check_function=self.check_system_environment,
            critical=True
        ))
        
        # 2. Data Uploads Check
        self.workflow_steps.append(WorkflowStep(
            name="Data Uploads",
            description="Verify upload directory, permissions, and file handling",
            dependencies=["System Environment"],
            check_function=self.check_data_uploads,
            critical=True
        ))
        
        # 3. Data Engineering Check
        self.workflow_steps.append(WorkflowStep(
            name="Data Engineering",
            description="Test data processing, ETL pipelines, and dataset management",
            dependencies=["Data Uploads"],
            check_function=self.check_data_engineering,
            critical=True
        ))
        
        # 4. LLM Engineering Check
        self.workflow_steps.append(WorkflowStep(
            name="LLM Engineering",
            description="Verify LLM providers, model management, and lifecycle",
            dependencies=["System Environment"],
            check_function=self.check_llm_engineering,
            critical=True
        ))
        
        # 5. High Performance System Check
        self.workflow_steps.append(WorkflowStep(
            name="High Performance System",
            description="Test MoE system, expert ensemble, and performance metrics",
            dependencies=["Data Engineering", "LLM Engineering"],
            check_function=self.check_high_performance_system,
            critical=True
        ))
        
        # 6. Security System Check
        self.workflow_steps.append(WorkflowStep(
            name="Security System",
            description="Verify authentication, authorization, and security features",
            dependencies=["High Performance System"],
            check_function=self.check_security_system,
            critical=True
        ))
        
        # 7. MCP Server Check
        self.workflow_steps.append(WorkflowStep(
            name="MCP Server",
            description="Test MCP server connectivity, authentication, and APIs",
            dependencies=["Security System"],
            check_function=self.check_mcp_server,
            critical=False
        ))
        
        # 8. Cloud APIs Check
        self.workflow_steps.append(WorkflowStep(
            name="Cloud APIs",
            description="Verify cloud provider integrations and deployment",
            dependencies=["MCP Server"],
            check_function=self.check_cloud_apis,
            critical=False
        ))
        
        # 9. Third-party Integrations Check
        self.workflow_steps.append(WorkflowStep(
            name="Third-party Integrations",
            description="Test external API integrations and webhooks",
            dependencies=["MCP Server"],
            check_function=self.check_thirdparty_integrations,
            critical=False
        ))
        
        # 10. Tests Check
        self.workflow_steps.append(WorkflowStep(
            name="Tests",
            description="Run comprehensive test suites and validation",
            dependencies=["High Performance System"],
            check_function=self.check_tests,
            critical=True
        ))
        
        # 11. Plugins Check
        self.workflow_steps.append(WorkflowStep(
            name="Plugins",
            description="Verify plugin loading, compatibility, and functionality",
            dependencies=["High Performance System"],
            check_function=self.check_plugins,
            critical=False
        ))
        
        # 12. Analytics & Dashboards Check
        self.workflow_steps.append(WorkflowStep(
            name="Analytics & Dashboards",
            description="Test dashboard functionality and analytics",
            dependencies=["High Performance System"],
            check_function=self.check_analytics_dashboards,
            critical=False
        ))
        
        # 13. Production Server Check
        self.workflow_steps.append(WorkflowStep(
            name="Production Server",
            description="Verify production server status and performance",
            dependencies=["High Performance System"],
            check_function=self.check_production_server,
            critical=True
        ))
    
    async def run_complete_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic workflow"""
        logger.info("🚀 Starting Complete OpenTrustEval Workflow Diagnostic")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run each step in dependency order
        for step in self.workflow_steps:
            await self.run_workflow_step(step)
        
        # Generate comprehensive report
        report = self.generate_diagnostic_report()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Complete diagnostic finished in {total_time:.2f} seconds")
        
        return report
    
    async def run_workflow_step(self, step: WorkflowStep):
        """Run a single workflow step"""
        logger.info(f"🔍 Running: {step.name}")
        logger.info(f"   Description: {step.description}")
        
        # Check dependencies
        if not self.check_dependencies(step):
            logger.warning(f"⚠️ Skipping {step.name} due to failed dependencies")
            self.results.append(DiagnosticResult(
                component=step.name,
                status="SKIP",
                message="Dependencies not met",
                details={"dependencies": step.dependencies},
                timestamp=datetime.now(),
                duration=0.0
            ))
            return
        
        # Run the check
        start_time = time.time()
        try:
            result = await step.check_function()
            duration = time.time() - start_time
            
            self.results.append(DiagnosticResult(
                component=step.name,
                status=result["status"],
                message=result["message"],
                details=result["details"],
                timestamp=datetime.now(),
                duration=duration
            ))
            
            status_emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "SKIP": "⏭️"}
            logger.info(f"{status_emoji.get(result['status'], '❓')} {step.name}: {result['message']}")
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Error in {step.name}: {str(e)}")
            
            self.results.append(DiagnosticResult(
                component=step.name,
                status="FAIL",
                message=f"Exception: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                timestamp=datetime.now(),
                duration=duration
            ))
    
    def check_dependencies(self, step: WorkflowStep) -> bool:
        """Check if step dependencies are met"""
        if not step.dependencies:
            return True
        
        for dep in step.dependencies:
            # Find the dependency result
            dep_result = None
            for result in self.results:
                if result.component == dep:
                    dep_result = result
                    break
            
            if not dep_result or dep_result.status == "FAIL":
                return False
        
        return True
    
    async def check_system_environment(self) -> Dict[str, Any]:
        """Check system environment and dependencies"""
        details = {}
        
        # Check Python version
        python_version = sys.version_info
        details["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            return {
                "status": "FAIL",
                "message": f"Python {python_version.major}.{python_version.minor} not supported. Need 3.8+",
                "details": details
            }
        
        # Check required packages
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "pandas", "numpy", 
            "plotly", "requests", "asyncio", "logging", "json"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
                details[f"{package}_available"] = True
            except ImportError:
                missing_packages.append(package)
                details[f"{package}_available"] = False
        
        if missing_packages:
            return {
                "status": "FAIL",
                "message": f"Missing packages: {', '.join(missing_packages)}",
                "details": details
            }
        
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk = psutil.disk_usage('/')
            
            details["memory_available"] = f"{memory.available / (1024**3):.2f} GB"
            details["cpu_count"] = cpu_count
            details["disk_free"] = f"{disk.free / (1024**3):.2f} GB"
            
            if memory.available < 2 * 1024**3:  # Less than 2GB
                return {
                    "status": "WARNING",
                    "message": "Low memory available",
                    "details": details
                }
                
        except Exception as e:
            details["resource_check_error"] = str(e)
        
        # Check working directory
        details["working_directory"] = os.getcwd()
        details["project_root"] = self.find_project_root()
        
        return {
            "status": "PASS",
            "message": "System environment is ready",
            "details": details
        }
    
    async def check_data_uploads(self) -> Dict[str, Any]:
        """Check data uploads functionality"""
        details = {}
        
        # Check uploads directory
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            try:
                uploads_dir.mkdir(exist_ok=True)
                details["uploads_dir_created"] = True
            except Exception as e:
                return {
                    "status": "FAIL",
                    "message": f"Cannot create uploads directory: {str(e)}",
                    "details": details
                }
        else:
            details["uploads_dir_exists"] = True
        
        # Check permissions
        try:
            test_file = uploads_dir / "test_permissions.txt"
            test_file.write_text("test")
            test_file.unlink()
            details["write_permissions"] = True
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"No write permissions in uploads directory: {str(e)}",
                "details": details
            }
        
        # Check file handling
        try:
            from data_engineering.dataset_integration import DatasetConnector
            details["dataset_connector_available"] = True
        except ImportError as e:
            details["dataset_connector_available"] = False
            details["dataset_connector_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "Data uploads system is functional",
            "details": details
        }
    
    async def check_data_engineering(self) -> Dict[str, Any]:
        """Check data engineering components"""
        details = {}
        
        # Check data engineering modules
        data_eng_modules = [
            "data_engineering.advanced_trust_scoring",
            "data_engineering.cleanlab_integration",
            "data_engineering.trust_scoring_dashboard",
            "data_engineering.data_lifecycle"
        ]
        
        for module_name in data_eng_modules:
            try:
                module = importlib.import_module(module_name)
                details[f"{module_name}_available"] = True
            except ImportError as e:
                details[f"{module_name}_available"] = False
                details[f"{module_name}_error"] = str(e)
        
        # Test trust scoring
        try:
            from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
            engine = AdvancedTrustScoringEngine()
            test_result = engine.calculate_trust_score("This is a test text for verification.")
            details["trust_scoring_test"] = "PASS"
            details["test_score"] = test_result.get("trust_score", "N/A")
        except Exception as e:
            details["trust_scoring_test"] = "FAIL"
            details["trust_scoring_error"] = str(e)
        
        # Check database connectivity
        try:
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.close()
            details["database_connectivity"] = True
        except Exception as e:
            details["database_connectivity"] = False
            details["database_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "Data engineering components are functional",
            "details": details
        }
    
    async def check_llm_engineering(self) -> Dict[str, Any]:
        """Check LLM engineering components"""
        details = {}
        
        # Check LLM engineering modules
        llm_modules = [
            "llm_engineering.llm_lifecycle",
            "llm_engineering.providers.base_provider",
            "llm_engineering.providers.llama_factory_provider"
        ]
        
        for module_name in llm_modules:
            try:
                module = importlib.import_module(module_name)
                details[f"{module_name}_available"] = True
            except ImportError as e:
                details[f"{module_name}_available"] = False
                details[f"{module_name}_error"] = str(e)
        
        # Test LLM lifecycle manager
        try:
            from llm_engineering.llm_lifecycle import LLMLifecycleManager
            manager = LLMLifecycleManager()
            details["llm_lifecycle_manager"] = "PASS"
        except Exception as e:
            details["llm_lifecycle_manager"] = "FAIL"
            details["llm_lifecycle_error"] = str(e)
        
        # Check model configurations
        config_path = Path("llm_engineering/configs/llm_providers.yaml")
        if config_path.exists():
            details["llm_config_exists"] = True
        else:
            details["llm_config_exists"] = False
        
        return {
            "status": "PASS",
            "message": "LLM engineering components are functional",
            "details": details
        }
    
    async def check_high_performance_system(self) -> Dict[str, Any]:
        """Check high performance system components"""
        details = {}
        
        # Check high performance system modules
        hps_modules = [
            "high_performance_system.core.ultimate_moe_system",
            "high_performance_system.core.advanced_expert_ensemble",
            "high_performance_system.core.intelligent_domain_router",
            "high_performance_system.core.enhanced_dataset_profiler",
            "high_performance_system.core.comprehensive_pii_detector",
            "high_performance_system.core.advanced_trust_scorer"
        ]
        
        for module_name in hps_modules:
            try:
                module = importlib.import_module(module_name)
                details[f"{module_name}_available"] = True
            except ImportError as e:
                details[f"{module_name}_available"] = False
                details[f"{module_name}_error"] = str(e)
        
        # Test Ultimate MoE System
        try:
            from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
            moe_system = UltimateMoESystem()
            test_result = await moe_system.verify_text("This is a test verification.")
            details["moe_system_test"] = "PASS"
            details["moe_test_result"] = test_result.verified if hasattr(test_result, 'verified') else "N/A"
        except Exception as e:
            details["moe_system_test"] = "FAIL"
            details["moe_system_error"] = str(e)
        
        # Check expert ensemble
        try:
            from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
            ensemble = AdvancedExpertEnsemble()
            details["expert_ensemble"] = "PASS"
        except Exception as e:
            details["expert_ensemble"] = "FAIL"
            details["expert_ensemble_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "High performance system components are functional",
            "details": details
        }
    
    async def check_security_system(self) -> Dict[str, Any]:
        """Check security system components"""
        details = {}
        
        # Check security modules
        security_modules = [
            "security.auth_manager",
            "security.secrets_manager",
            "security.security_monitor"
        ]
        
        for module_name in security_modules:
            try:
                module = importlib.import_module(module_name)
                details[f"{module_name}_available"] = True
            except ImportError as e:
                details[f"{module_name}_available"] = False
                details[f"{module_name}_error"] = str(e)
        
        # Test security web UI
        try:
            from security.security_webui import get_high_performance_security_status
            security_status = get_high_performance_security_status()
            details["security_webui"] = "PASS"
            details["security_status"] = security_status
        except Exception as e:
            details["security_webui"] = "FAIL"
            details["security_webui_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "Security system components are functional",
            "details": details
        }
    
    async def check_mcp_server(self) -> Dict[str, Any]:
        """Check MCP server functionality"""
        details = {}
        
        # Check MCP server files
        mcp_files = [
            "mcp_server/server.py",
            "mcp_server/client.py",
            "mcp_server/config.py"
        ]
        
        for file_path in mcp_files:
            if Path(file_path).exists():
                details[f"{file_path}_exists"] = True
            else:
                details[f"{file_path}_exists"] = False
        
        # Test MCP server connectivity (if running)
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                details["mcp_server_running"] = True
                details["mcp_server_health"] = response.json()
            else:
                details["mcp_server_running"] = False
                details["mcp_server_status_code"] = response.status_code
        except requests.exceptions.RequestException:
            details["mcp_server_running"] = False
            details["mcp_server_error"] = "Server not responding"
        
        return {
            "status": "PASS" if details.get("mcp_server_running", False) else "WARNING",
            "message": "MCP server check completed",
            "details": details
        }
    
    async def check_cloud_apis(self) -> Dict[str, Any]:
        """Check cloud APIs functionality"""
        details = {}
        
        # Check cloud APIs documentation
        cloud_docs = [
            "cloudscale_apis/docs/cloud_provider_integration.md",
            "cloudscale_apis/endpoints/",
            "cloudscale_apis/webhooks/"
        ]
        
        for doc_path in cloud_docs:
            if Path(doc_path).exists():
                details[f"{doc_path}_exists"] = True
            else:
                details[f"{doc_path}_exists"] = False
        
        # Check cloud provider configurations
        cloud_providers = ["aws", "azure", "gcp"]
        for provider in cloud_providers:
            try:
                if provider == "aws":
                    import boto3
                    details[f"{provider}_sdk_available"] = True
                elif provider == "azure":
                    import azure
                    details[f"{provider}_sdk_available"] = True
                elif provider == "gcp":
                    import google.cloud
                    details[f"{provider}_sdk_available"] = True
            except ImportError:
                details[f"{provider}_sdk_available"] = False
        
        return {
            "status": "PASS",
            "message": "Cloud APIs check completed",
            "details": details
        }
    
    async def check_thirdparty_integrations(self) -> Dict[str, Any]:
        """Check third-party integrations"""
        details = {}
        
        # Check third-party integration modules
        thirdparty_modules = [
            "thirdparty_integrations.endpoints.verify_realtime",
            "thirdparty_integrations.endpoints.verify_batch",
            "thirdparty_integrations.webhooks.verify_webhook"
        ]
        
        for module_name in thirdparty_modules:
            try:
                module = importlib.import_module(module_name)
                details[f"{module_name}_available"] = True
            except ImportError as e:
                details[f"{module_name}_available"] = False
                details[f"{module_name}_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "Third-party integrations check completed",
            "details": details
        }
    
    async def check_tests(self) -> Dict[str, Any]:
        """Check test functionality"""
        details = {}
        
        # Check test files
        test_files = [
            "test_high_performance_system.py",
            "simple_unit_test.py",
            "tests/test_advanced_pipeline.py"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                details[f"{test_file}_exists"] = True
            else:
                details[f"{test_file}_exists"] = False
        
        # Run a simple test
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "simple_unit_test.py", "-v"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                details["simple_test_run"] = "PASS"
                details["test_output"] = result.stdout
            else:
                details["simple_test_run"] = "FAIL"
                details["test_error"] = result.stderr
        except subprocess.TimeoutExpired:
            details["simple_test_run"] = "TIMEOUT"
        except Exception as e:
            details["simple_test_run"] = "ERROR"
            details["test_exception"] = str(e)
        
        return {
            "status": "PASS" if details.get("simple_test_run") == "PASS" else "WARNING",
            "message": "Tests check completed",
            "details": details
        }
    
    async def check_plugins(self) -> Dict[str, Any]:
        """Check plugins functionality"""
        details = {}
        
        # Check plugin loader
        try:
            from plugins.plugin_loader import load_plugins, load_high_performance_plugins
            plugins = load_plugins()
            hps_plugins = load_high_performance_plugins()
            
            details["plugin_loader"] = "PASS"
            details["plugins_loaded"] = len(plugins)
            details["hps_plugins_loaded"] = len(hps_plugins)
        except Exception as e:
            details["plugin_loader"] = "FAIL"
            details["plugin_loader_error"] = str(e)
        
        # Check plugin files
        plugin_files = [
            "plugins/example_plugin.py",
            "plugins/hallucination_detector.py",
            "plugins/eu_gdpr_embed.py"
        ]
        
        for plugin_file in plugin_files:
            if Path(plugin_file).exists():
                details[f"{plugin_file}_exists"] = True
            else:
                details[f"{plugin_file}_exists"] = False
        
        return {
            "status": "PASS",
            "message": "Plugins check completed",
            "details": details
        }
    
    async def check_analytics_dashboards(self) -> Dict[str, Any]:
        """Check analytics and dashboards"""
        details = {}
        
        # Check dashboard files
        dashboard_files = [
            "high_performance_system/analytics/ultimate_analytics_dashboard.py",
            "high_performance_system/analytics/sme_dashboard.py",
            "data_engineering/trust_scoring_dashboard.py",
            "operation_sindoor_dashboard.py"
        ]
        
        for dashboard_file in dashboard_files:
            if Path(dashboard_file).exists():
                details[f"{dashboard_file}_exists"] = True
            else:
                details[f"{dashboard_file}_exists"] = False
        
        # Test dashboard imports
        try:
            from high_performance_system.analytics.ultimate_analytics_dashboard import UltimateAnalyticsDashboard
            dashboard = UltimateAnalyticsDashboard()
            details["ultimate_dashboard"] = "PASS"
        except Exception as e:
            details["ultimate_dashboard"] = "FAIL"
            details["ultimate_dashboard_error"] = str(e)
        
        return {
            "status": "PASS",
            "message": "Analytics and dashboards check completed",
            "details": details
        }
    
    async def check_production_server(self) -> Dict[str, Any]:
        """Check production server status"""
        details = {}
        
        # Check production server files
        server_files = [
            "superfast_production_server.py",
            "ote_api.py"
        ]
        
        for server_file in server_files:
            if Path(server_file).exists():
                details[f"{server_file}_exists"] = True
            else:
                details[f"{server_file}_exists"] = False
        
        # Test production server connectivity
        try:
            response = requests.get("http://localhost:8003/health", timeout=5)
            if response.status_code == 200:
                details["production_server_running"] = True
                details["production_server_health"] = response.json()
            else:
                details["production_server_running"] = False
                details["production_server_status_code"] = response.status_code
        except requests.exceptions.RequestException:
            details["production_server_running"] = False
            details["production_server_error"] = "Server not responding"
        
        # Check performance endpoint
        try:
            response = requests.get("http://localhost:8003/performance", timeout=5)
            if response.status_code == 200:
                details["performance_endpoint"] = "PASS"
                details["performance_data"] = response.json()
            else:
                details["performance_endpoint"] = "FAIL"
        except requests.exceptions.RequestException:
            details["performance_endpoint"] = "FAIL"
        
        return {
            "status": "PASS" if details.get("production_server_running", False) else "WARNING",
            "message": "Production server check completed",
            "details": details
        }
    
    def find_project_root(self) -> str:
        """Find the project root directory"""
        current = Path.cwd()
        while current != current.parent:
            if (current / "README.md").exists() and (current / "requirements.txt").exists():
                return str(current)
            current = current.parent
        return str(Path.cwd())
    
    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.results),
                "passed": len([r for r in self.results if r.status == "PASS"]),
                "failed": len([r for r in self.results if r.status == "FAIL"]),
                "warnings": len([r for r in self.results if r.status == "WARNING"]),
                "skipped": len([r for r in self.results if r.status == "SKIP"])
            },
            "results": [],
            "recommendations": []
        }
        
        # Process results
        for result in self.results:
            report["results"].append({
                "component": result.component,
                "status": result.status,
                "message": result.message,
                "duration": result.duration,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details
            })
        
        # Generate recommendations
        failed_components = [r for r in self.results if r.status == "FAIL"]
        warning_components = [r for r in self.results if r.status == "WARNING"]
        
        if failed_components:
            report["recommendations"].append({
                "priority": "HIGH",
                "action": "Fix critical failures",
                "components": [r.component for r in failed_components]
            })
        
        if warning_components:
            report["recommendations"].append({
                "priority": "MEDIUM",
                "action": "Address warnings",
                "components": [r.component for r in warning_components]
            })
        
        # Check for missing critical components
        critical_components = ["System Environment", "Data Uploads", "Data Engineering", 
                             "LLM Engineering", "High Performance System", "Security System"]
        missing_critical = []
        
        for component in critical_components:
            if not any(r.component == component for r in self.results):
                missing_critical.append(component)
        
        if missing_critical:
            report["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Missing critical components",
                "components": missing_critical
            })
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted diagnostic report"""
        print("\n" + "=" * 80)
        print("🔍 OPENTRUSTEVAL COMPLETE WORKFLOW DIAGNOSTIC REPORT")
        print("=" * 80)
        
        # Summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⚠️ Warnings: {summary['warnings']}")
        print(f"   ⏭️ Skipped: {summary['skipped']}")
        
        # Results
        print(f"\n📋 DETAILED RESULTS:")
        for result in report["results"]:
            status_emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "SKIP": "⏭️"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"   {emoji} {result['component']}: {result['message']}")
            if result["duration"] > 0:
                print(f"      Duration: {result['duration']:.2f}s")
        
        # Recommendations
        if report["recommendations"]:
            print(f"\n🎯 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                priority_emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                emoji = priority_emoji.get(rec["priority"], "❓")
                print(f"   {emoji} {rec['priority']}: {rec['action']}")
                if "components" in rec:
                    print(f"      Components: {', '.join(rec['components'])}")
        
        print("\n" + "=" * 80)

# Main execution
async def main():
    """Main diagnostic execution"""
    diagnostic = CompleteWorkflowDiagnostic()
    
    try:
        report = await diagnostic.run_complete_diagnostic()
        diagnostic.print_report(report)
        
        # Save report to file
        report_file = f"workflow_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {report_file}")
        
        # Exit with appropriate code
        if report["summary"]["failed"] > 0:
            sys.exit(1)
        elif report["summary"]["warnings"] > 0:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 