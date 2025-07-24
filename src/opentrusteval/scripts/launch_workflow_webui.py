#!/usr/bin/env python3
"""
OpenTrustEval Unified Workflow Web UI Launcher
Launches the comprehensive unified web interface that integrates all WebUIs
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are available")
    return True

def check_system_status():
    """Check system status"""
    print("🔍 Checking system status...")
    
    # Check if production server is running
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            print("✅ Production server is running")
        else:
            print("⚠️ Production server is not running")
    except:
        print("⚠️ Production server is not running")
    
    # Check if MCP server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ MCP server is running")
        else:
            print("⚠️ MCP server is not running")
    except:
        print("⚠️ MCP server is not running")
    
    # Check uploads directory
    uploads_exists = Path("uploads").exists()
    if uploads_exists:
        print("✅ Uploads directory exists")
    else:
        print("⚠️ Uploads directory missing")
    
    # Check system readiness
    print("✅ System ready for advanced research operations")

def launch_webui():
    """Launch the unified workflow web UI"""
    print("🚀 Launching web interface...")
    
    # Check if workflow_webui.py exists
    if not Path("workflow_webui.py").exists():
        print("❌ workflow_webui.py not found")
        return False
    
    print("🚀 Launching OpenTrustEval Unified Workflow Web UI...")
    print("🌐 Starting web interface on port 8501...")
    
    try:
        # Launch Streamlit app
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "workflow_webui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the server is running
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ Web UI launched successfully!")
                print("🌐 Access URL: http://localhost:8501")
                print("🎯 Web UI Features:")
                print("   🏠 Dashboard - System overview and quick actions")
                print("   🔍 System Diagnostic - Comprehensive health checks")
                print("   🔧 Problem Resolution - Interactive issue fixing")
                print("   🚀 Service Management - Start/stop services")
                print("   📊 Analytics & Monitoring - Performance tracking")
                print("   🧪 Testing & Validation - Run test suites")
                print("   📋 Reports & Logs - View system reports")
                print("   ⚙️ Configuration - System settings")
                print("   📁 Dataset Management - Upload, validate, visualize datasets")
                print("   🤖 LLM Model Manager - Model management and tuning")
                print("   🔒 Security Management - User management and security monitoring")
                print("🔄 Web UI is running. Press Ctrl+C to stop.")
                
                return process
            else:
                print("❌ Web UI failed to start properly")
                return False
        except requests.exceptions.RequestException:
            print("❌ Web UI failed to start properly")
            return False
            
    except Exception as e:
        print(f"❌ Error launching web UI: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 OpenTrustEval Unified Workflow Web UI Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check system status
    check_system_status()
    
    # Launch web UI
    process = launch_webui()
    
    if process:
        try:
            # Keep the process running
            process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping web UI...")
            process.terminate()
            process.wait()
            print("✅ Web UI stopped")

if __name__ == "__main__":
    main() 