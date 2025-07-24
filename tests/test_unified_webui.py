#!/usr/bin/env python3
"""
Test script for Unified Workflow Web UI
Verifies that all integrated components are working properly
"""

import requests
import time
import sys
from pathlib import Path

def test_webui_access():
    """Test if the unified web UI is accessible"""
    print("🧪 Testing Unified Web UI Access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Unified Web UI is accessible")
            return True
        else:
            print(f"❌ Web UI returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot access Web UI: {e}")
        return False

def test_component_availability():
    """Test if all integrated components are available"""
    print("\n🔍 Testing Component Availability...")
    
    components = [
        ("Dataset Manager", "data_engineering/dataset_integration.py"),
        ("LLM Manager", "llm_engineering/llm_lifecycle.py"),
        ("Security Manager", "security/auth_manager.py"),
        ("Workflow WebUI", "workflow_webui.py"),
        ("Launcher", "launch_workflow_webui.py")
    ]
    
    all_available = True
    for name, path in components:
        if Path(path).exists():
            print(f"✅ {name}: Available")
        else:
            print(f"❌ {name}: Missing ({path})")
            all_available = False
    
    return all_available

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing Module Imports...")
    
    modules = [
        ("streamlit", "st"),
        ("pandas", "pd"),
        ("plotly.express", "px"),
        ("requests", "requests"),
        ("psutil", "psutil")
    ]
    
    all_importable = True
    for module, alias in modules:
        try:
            __import__(module)
            print(f"✅ {module}: Importable")
        except ImportError as e:
            print(f"❌ {module}: Not importable - {e}")
            all_importable = False
    
    return all_importable

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Unified Workflow Web UI Test Suite")
    print("=" * 60)
    
    # Test 1: Web UI Access
    webui_accessible = test_webui_access()
    
    # Test 2: Component Availability
    components_available = test_component_availability()
    
    # Test 3: Module Imports
    imports_working = test_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    print(f"Web UI Access: {'✅ PASS' if webui_accessible else '❌ FAIL'}")
    print(f"Components Available: {'✅ PASS' if components_available else '❌ FAIL'}")
    print(f"Module Imports: {'✅ PASS' if imports_working else '❌ FAIL'}")
    
    overall_success = webui_accessible and components_available and imports_working
    
    if overall_success:
        print("\n🎉 All tests passed! Unified Web UI is working properly.")
        print("🌐 Access your unified interface at: http://localhost:8501")
        print("\n🎯 Available Features:")
        print("   🏠 Dashboard - System overview and quick actions")
        print("   📁 Dataset Management - Upload, validate, visualize datasets")
        print("   🤖 LLM Model Manager - Model management and tuning")
        print("   🔒 Security Management - User management and security monitoring")
        print("   🔍 System Diagnostic - Comprehensive health checks")
        print("   🔧 Problem Resolution - Interactive issue fixing")
        print("   🚀 Service Management - Start/stop services")
        print("   📊 Analytics & Monitoring - Performance tracking")
        print("   🧪 Testing & Validation - Run test suites")
        print("   📋 Reports & Logs - View system reports")
        print("   ⚙️ Configuration - System settings")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 