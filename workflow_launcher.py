#!/usr/bin/env python3
"""
Complete Workflow Launcher for OpenTrustEval
Unified interface for diagnostics, problem resolution, and system management
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

class WorkflowLauncher:
    """Complete workflow launcher for OpenTrustEval"""
    
    def __init__(self):
        self.diagnostic_script = "complete_workflow_diagnostic.py"
        self.resolver_script = "workflow_problem_resolver.py"
        self.dashboard_launcher = "launch_operation_sindoor_dashboard.py"
        self.production_server = "superfast_production_server.py"
        
    def show_menu(self):
        """Show main workflow menu"""
        print("\n" + "=" * 80)
        print("🚀 OPENTRUSTEVAL COMPLETE WORKFLOW LAUNCHER")
        print("=" * 80)
        print("🎯 Comprehensive system management and problem resolution")
        print("=" * 80)
        
        print("\n📋 Available Operations:")
        print("   1. 🔍 Run Complete System Diagnostic")
        print("   2. 🔧 Resolve Problems (Interactive)")
        print("   3. 🚀 Start Production Server")
        print("   4. 📊 Launch Dashboards")
        print("   5. 🧪 Run Tests")
        print("   6. 🔄 Full System Check & Fix")
        print("   7. 📈 System Status Overview")
        print("   8. 🛠️ Component-Specific Checks")
        print("   9. 📋 View Recent Reports")
        print("   0. ❌ Exit")
        
        print("\n💡 Quick Actions:")
        print("   d - Quick diagnostic")
        print("   s - Start server")
        print("   t - Run tests")
        print("   h - Show help")
    
    def run_diagnostic(self, quick: bool = False) -> bool:
        """Run system diagnostic"""
        print(f"\n🔍 Running {'quick' if quick else 'complete'} system diagnostic...")
        
        try:
            if quick:
                # Quick diagnostic using subprocess
                result = subprocess.run([
                    sys.executable, self.diagnostic_script, "--quick"
                ], capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run([
                    sys.executable, self.diagnostic_script
                ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Diagnostic completed successfully")
                print(result.stdout)
                return True
            else:
                print("❌ Diagnostic failed")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Diagnostic timed out")
            return False
        except Exception as e:
            print(f"❌ Diagnostic error: {str(e)}")
            return False
    
    def resolve_problems(self) -> bool:
        """Run problem resolver"""
        print("\n🔧 Running problem resolver...")
        
        try:
            result = subprocess.run([
                sys.executable, self.resolver_script
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Problem resolution completed")
                print(result.stdout)
                return True
            else:
                print("❌ Problem resolution failed")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Problem resolution timed out")
            return False
        except Exception as e:
            print(f"❌ Problem resolution error: {str(e)}")
            return False
    
    def start_production_server(self) -> bool:
        """Start production server"""
        print("\n🚀 Starting production server...")
        
        try:
            # Check if server is already running
            import requests
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Production server is already running")
                    return True
            except:
                pass
            
            # Start server in background
            process = subprocess.Popen([
                sys.executable, self.production_server
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if server started successfully
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Production server started successfully")
                    print(f"   URL: http://localhost:8003")
                    print(f"   Health: {response.json()}")
                    return True
                else:
                    print(f"❌ Server started but health check failed: {response.status_code}")
                    return False
            except:
                print("❌ Server failed to start or health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {str(e)}")
            return False
    
    def launch_dashboards(self) -> bool:
        """Launch dashboards"""
        print("\n📊 Launching dashboards...")
        
        try:
            result = subprocess.run([
                sys.executable, self.dashboard_launcher
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Dashboard launcher completed")
                print(result.stdout)
                return True
            else:
                print("❌ Dashboard launcher failed")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Dashboard launcher timed out")
            return False
        except Exception as e:
            print(f"❌ Dashboard launcher error: {str(e)}")
            return False
    
    def run_tests(self) -> bool:
        """Run system tests"""
        print("\n🧪 Running system tests...")
        
        test_files = [
            "simple_unit_test.py",
            "test_high_performance_system.py"
        ]
        
        success = True
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"   Running {test_file}...")
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v"
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        print(f"   ✅ {test_file} passed")
                    else:
                        print(f"   ❌ {test_file} failed")
                        print(f"      {result.stderr}")
                        success = False
                except Exception as e:
                    print(f"   ❌ {test_file} error: {str(e)}")
                    success = False
            else:
                print(f"   ⚠️ {test_file} not found")
        
        if success:
            print("✅ All tests completed successfully")
        else:
            print("❌ Some tests failed")
        
        return success
    
    def full_system_check_and_fix(self) -> bool:
        """Run complete system check and fix workflow"""
        print("\n🔄 Running full system check and fix workflow...")
        
        # Step 1: Run diagnostic
        print("\n📋 Step 1: Running diagnostic...")
        if not self.run_diagnostic():
            print("❌ Diagnostic failed, stopping workflow")
            return False
        
        # Step 2: Resolve problems
        print("\n📋 Step 2: Resolving problems...")
        if not self.resolve_problems():
            print("⚠️ Problem resolution had issues, continuing...")
        
        # Step 3: Run tests
        print("\n📋 Step 3: Running tests...")
        if not self.run_tests():
            print("⚠️ Some tests failed, continuing...")
        
        # Step 4: Start server
        print("\n📋 Step 4: Starting production server...")
        if not self.start_production_server():
            print("⚠️ Server start had issues")
        
        print("\n✅ Full system check and fix workflow completed")
        return True
    
    def system_status_overview(self) -> Dict[str, Any]:
        """Get system status overview"""
        print("\n📈 System Status Overview")
        print("=" * 40)
        
        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "components": {}
        }
        
        # Check production server
        try:
            import requests
            response = requests.get("http://localhost:8003/health", timeout=5)
            status["components"]["production_server"] = {
                "status": "RUNNING" if response.status_code == 200 else "ERROR",
                "response_time": response.elapsed.total_seconds(),
                "details": response.json() if response.status_code == 200 else None
            }
        except:
            status["components"]["production_server"] = {
                "status": "NOT_RUNNING",
                "response_time": None,
                "details": None
            }
        
        # Check MCP server
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            status["components"]["mcp_server"] = {
                "status": "RUNNING" if response.status_code == 200 else "ERROR",
                "response_time": response.elapsed.total_seconds(),
                "details": response.json() if response.status_code == 200 else None
            }
        except:
            status["components"]["mcp_server"] = {
                "status": "NOT_RUNNING",
                "response_time": None,
                "details": None
            }
        
        # Check file system
        status["components"]["file_system"] = {
            "uploads_dir": Path("uploads").exists(),
            "data_engineering": Path("data_engineering").exists(),
            "high_performance_system": Path("high_performance_system").exists(),
            "llm_engineering": Path("llm_engineering").exists(),
            "security": Path("security").exists(),
            "mcp_server": Path("mcp_server").exists()
        }
        
        # Check Python environment
        status["components"]["python_environment"] = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "working_directory": os.getcwd()
        }
        
        # Print status
        for component, info in status["components"].items():
            if isinstance(info, dict) and "status" in info:
                status_emoji = {"RUNNING": "✅", "ERROR": "❌", "NOT_RUNNING": "⏸️"}
                emoji = status_emoji.get(info["status"], "❓")
                print(f"   {emoji} {component}: {info['status']}")
            elif isinstance(info, dict):
                print(f"   📁 {component}:")
                for key, value in info.items():
                    status_emoji = "✅" if value else "❌"
                    print(f"      {status_emoji} {key}: {value}")
        
        return status
    
    def component_specific_checks(self):
        """Run component-specific checks"""
        print("\n🛠️ Component-Specific Checks")
        print("=" * 40)
        
        components = [
            ("Data Engineering", "data_engineering/"),
            ("LLM Engineering", "llm_engineering/"),
            ("High Performance System", "high_performance_system/"),
            ("Security", "security/"),
            ("MCP Server", "mcp_server/"),
            ("Plugins", "plugins/"),
            ("Tests", "tests/")
        ]
        
        for name, path in components:
            print(f"\n🔍 Checking {name}...")
            
            if Path(path).exists():
                print(f"   ✅ Directory exists: {path}")
                
                # Count files
                files = list(Path(path).rglob("*.py"))
                print(f"   📄 Python files: {len(files)}")
                
                # Check for key files
                key_files = {
                    "data_engineering": ["advanced_trust_scoring.py", "trust_scoring_dashboard.py"],
                    "llm_engineering": ["llm_lifecycle.py"],
                    "high_performance_system": ["core/ultimate_moe_system.py"],
                    "security": ["security_webui.py"],
                    "mcp_server": ["server.py"],
                    "plugins": ["plugin_loader.py"],
                    "tests": ["test_advanced_pipeline.py"]
                }
                
                if name.lower().replace(" ", "_") in key_files:
                    for key_file in key_files[name.lower().replace(" ", "_")]:
                        if Path(path) / key_file in files:
                            print(f"   ✅ Key file: {key_file}")
                        else:
                            print(f"   ❌ Missing key file: {key_file}")
            else:
                print(f"   ❌ Directory missing: {path}")
    
    def view_recent_reports(self):
        """View recent diagnostic reports"""
        print("\n📋 Recent Diagnostic Reports")
        print("=" * 40)
        
        report_files = [f for f in os.listdir('.') if f.startswith('workflow_diagnostic_report_') and f.endswith('.json')]
        
        if not report_files:
            print("   No diagnostic reports found")
            return
        
        # Sort by modification time (newest first)
        report_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for i, report_file in enumerate(report_files[:5], 1):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(report_file)))
            size = os.path.getsize(report_file)
            
            print(f"   {i}. {report_file}")
            print(f"      Modified: {mtime}")
            print(f"      Size: {size} bytes")
            
            # Show summary if possible
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    summary = report.get("summary", {})
                    print(f"      Summary: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed")
            except:
                print(f"      Summary: Unable to read")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 OpenTrustEval Workflow Launcher Help")
        print("=" * 50)
        
        print("\n🎯 Purpose:")
        print("   This launcher provides a unified interface for managing the complete")
        print("   OpenTrustEval system, including diagnostics, problem resolution,")
        print("   and system monitoring.")
        
        print("\n🔧 Available Operations:")
        print("   1. Complete System Diagnostic - Comprehensive health check")
        print("   2. Problem Resolution - Interactive fix for issues")
        print("   3. Production Server - Start the main API server")
        print("   4. Dashboards - Launch analytics and monitoring dashboards")
        print("   5. Tests - Run system test suites")
        print("   6. Full Check & Fix - Complete workflow automation")
        print("   7. Status Overview - Current system status")
        print("   8. Component Checks - Specific component verification")
        print("   9. Recent Reports - View diagnostic history")
        
        print("\n⚡ Quick Commands:")
        print("   d - Quick diagnostic")
        print("   s - Start server")
        print("   t - Run tests")
        print("   h - Show this help")
        
        print("\n📁 Key Files:")
        print("   complete_workflow_diagnostic.py - Main diagnostic script")
        print("   workflow_problem_resolver.py - Problem resolution script")
        print("   superfast_production_server.py - Production server")
        print("   launch_operation_sindoor_dashboard.py - Dashboard launcher")
        
        print("\n🔗 Useful URLs:")
        print("   Production Server: http://localhost:8003")
        print("   MCP Server: http://localhost:8000")
        print("   Dashboards: http://localhost:8501-8504")
    
    def run(self):
        """Main launcher loop"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\n🎯 Select operation (0-9, or quick command): ").strip().lower()
                
                if choice == "0" or choice == "exit":
                    print("\n👋 Goodbye!")
                    break
                
                elif choice == "1" or choice == "d":
                    self.run_diagnostic()
                
                elif choice == "2":
                    self.resolve_problems()
                
                elif choice == "3" or choice == "s":
                    self.start_production_server()
                
                elif choice == "4":
                    self.launch_dashboards()
                
                elif choice == "5" or choice == "t":
                    self.run_tests()
                
                elif choice == "6":
                    self.full_system_check_and_fix()
                
                elif choice == "7":
                    self.system_status_overview()
                
                elif choice == "8":
                    self.component_specific_checks()
                
                elif choice == "9":
                    self.view_recent_reports()
                
                elif choice == "h" or choice == "help":
                    self.show_help()
                
                else:
                    print("❌ Invalid choice. Please select 0-9 or a quick command.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                input("Press Enter to continue...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OpenTrustEval Workflow Launcher")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic")
    parser.add_argument("--resolve", action="store_true", help="Resolve problems")
    parser.add_argument("--server", action="store_true", help="Start server")
    parser.add_argument("--tests", action="store_true", help="Run tests")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--full", action="store_true", help="Full check and fix")
    
    args = parser.parse_args()
    
    launcher = WorkflowLauncher()
    
    if args.diagnostic:
        launcher.run_diagnostic()
    elif args.resolve:
        launcher.resolve_problems()
    elif args.server:
        launcher.start_production_server()
    elif args.tests:
        launcher.run_tests()
    elif args.status:
        launcher.system_status_overview()
    elif args.full:
        launcher.full_system_check_and_fix()
    else:
        launcher.run()

if __name__ == "__main__":
    main() 