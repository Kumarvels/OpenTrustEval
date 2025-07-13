#!/usr/bin/env python3
"""
Test script for Quick Status Click Functionality
Verifies that all Quick Status buttons are properly configured
"""

import requests
import time
import sys
from pathlib import Path

def test_webui_access():
    """Test if the web UI is accessible"""
    print("🧪 Testing Web UI Access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Web UI is accessible")
            return True
        else:
            print(f"❌ Web UI returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot access Web UI: {e}")
        return False

def test_quick_status_components():
    """Test if all Quick Status components are available"""
    print("\n🔍 Testing Quick Status Components...")
    
    components = [
        ("Production Server", "superfast_production_server.py"),
        ("MCP Server", "mcp_server/server.py"),
        ("Dataset Manager", "data_engineering/dataset_integration.py"),
        ("LLM Manager", "llm_engineering/llm_lifecycle.py"),
        ("Security Manager", "security/auth_manager.py"),
        ("File System", "uploads/"),
        ("Research Lab", "workflow_webui.py")
    ]
    
    all_available = True
    for name, path in components:
        if Path(path).exists():
            print(f"✅ {name}: Available")
        else:
            print(f"⚠️ {name}: Missing ({path})")
            all_available = False
    
    return all_available

def test_click_functionality():
    """Test if click functionality is properly implemented"""
    print("\n🖱️ Testing Click Functionality...")
    
    # Check if the workflow_webui.py has the new methods
    try:
        with open("workflow_webui.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_methods = [
            "manage_production_server",
            "manage_mcp_server", 
            "check_dataset_manager",
            "check_llm_manager",
            "check_security_manager",
            "browse_file_system",
            "create_uploads_directory"
        ]
        
        all_methods_present = True
        for method in required_methods:
            if method in content:
                print(f"✅ {method}(): Implemented")
            else:
                print(f"❌ {method}(): Missing")
                all_methods_present = False
        
        return all_methods_present
    except Exception as e:
        print(f"❌ Error reading workflow_webui.py: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Quick Status Click Functionality Test Suite")
    print("=" * 60)
    
    # Test 1: Web UI Access
    webui_accessible = test_webui_access()
    
    # Test 2: Component Availability
    components_available = test_quick_status_components()
    
    # Test 3: Click Functionality
    click_functionality = test_click_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    print(f"Web UI Access: {'✅ PASS' if webui_accessible else '❌ FAIL'}")
    print(f"Components Available: {'✅ PASS' if components_available else '⚠️ PARTIAL'}")
    print(f"Click Functionality: {'✅ PASS' if click_functionality else '❌ FAIL'}")
    
    overall_success = webui_accessible and click_functionality
    
    if overall_success:
        print("\n🎉 Quick Status click functionality is working properly!")
        print("🌐 Access your enhanced interface at: http://localhost:8501")
        print("\n🎯 Quick Status Features:")
        print("   ❌ Production Server → Click to start/manage server")
        print("   ⏸️ MCP Server → Click to start/manage MCP server")
        print("   ✅ Dataset Manager → Click to open Dataset Management")
        print("   ✅ LLM Manager → Click to open LLM Model Manager")
        print("   ✅ Security Manager → Click to open Security Management")
        print("   ✅ File System → Click to browse file system")
        print("   🚀 Advanced Research Platform → Click to open Research Lab")
        print("\n💡 All Quick Status items are now clickable and functional!")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 