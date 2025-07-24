#!/usr/bin/env python3
"""
Operation Sindoor Dashboard Launcher
Integrates with existing OpenTrustEval dashboard infrastructure
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Dependencies installed successfully!")
    else:
        print("✅ All dependencies are available")

def check_report_file():
    """Check if the Operation Sindoor report file exists"""
    report_file = "operation_sindoor_test_report_20250713_134117.json"
    
    if os.path.exists(report_file):
        print(f"✅ Found report file: {report_file}")
        return True
    else:
        print(f"❌ Report file not found: {report_file}")
        print("Please ensure the Operation Sindoor test report exists in the current directory.")
        return False

def launch_dashboard(port=8503):
    """Launch the Operation Sindoor dashboard"""
    print(f"🚀 Launching Operation Sindoor Dashboard on port {port}...")
    
    # Check if port is available
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"⚠️  Port {port} is already in use. Trying port {port + 1}...")
            port += 1
    except:
        pass
    
    # Launch the dashboard
    dashboard_file = "operation_sindoor_dashboard.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        # Start the Streamlit dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            dashboard_file,
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"📊 Starting dashboard with command: {' '.join(cmd)}")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the dashboard
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def launch_all_dashboards():
    """Launch all available dashboards"""
    print("🎯 OpenTrustEval Dashboard Suite")
    print("=" * 50)
    
    dashboards = [
        {
            "name": "Operation Sindoor Dashboard",
            "file": "operation_sindoor_dashboard.py",
            "port": 8503,
            "description": "Specialized dashboard for Operation Sindoor analysis"
        },
        {
            "name": "Ultimate Analytics Dashboard", 
            "file": "high_performance_system/analytics/ultimate_analytics_dashboard.py",
            "port": 8501,
            "description": "Comprehensive MoE analytics dashboard"
        },
        {
            "name": "SME Dashboard",
            "file": "high_performance_system/analytics/sme_dashboard.py", 
            "port": 8502,
            "description": "Subject Matter Expert dashboard"
        },
        {
            "name": "Trust Scoring Dashboard",
            "file": "data_engineering/trust_scoring_dashboard.py",
            "port": 8504,
            "description": "Trust scoring system dashboard"
        }
    ]
    
    print("Available dashboards:")
    for i, dashboard in enumerate(dashboards, 1):
        status = "✅" if os.path.exists(dashboard["file"]) else "❌"
        print(f"{i}. {status} {dashboard['name']} (Port {dashboard['port']})")
        print(f"   {dashboard['description']}")
        print()
    
    choice = input("Select dashboard to launch (1-4) or 'all' for all dashboards: ").strip()
    
    if choice.lower() == 'all':
        print("🚀 Launching all dashboards...")
        for dashboard in dashboards:
            if os.path.exists(dashboard["file"]):
                print(f"📊 Launching {dashboard['name']} on port {dashboard['port']}...")
                # Launch each dashboard in a separate process
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run",
                    dashboard["file"],
                    "--server.port", str(dashboard["port"]),
                    "--server.address", "localhost",
                    "--browser.gatherUsageStats", "false"
                ])
                time.sleep(2)  # Small delay between launches
        
        print("✅ All dashboards launched!")
        print("\nDashboard URLs:")
        for dashboard in dashboards:
            if os.path.exists(dashboard["file"]):
                print(f"• {dashboard['name']}: http://localhost:{dashboard['port']}")
        
        # Open Operation Sindoor dashboard by default
        webbrowser.open(f"http://localhost:8503")
        
    elif choice.isdigit() and 1 <= int(choice) <= len(dashboards):
        selected = dashboards[int(choice) - 1]
        if os.path.exists(selected["file"]):
            launch_dashboard(selected["port"])
        else:
            print(f"❌ Dashboard file not found: {selected['file']}")
    else:
        print("❌ Invalid choice")

def main():
    """Main function"""
    print("🎯 Operation Sindoor Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Check report file
    if not check_report_file():
        print("\n💡 You can still launch other dashboards or run tests to generate new reports.")
    
    print("\nOptions:")
    print("1. Launch Operation Sindoor Dashboard only")
    print("2. Launch all available dashboards")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        launch_dashboard()
    elif choice == "2":
        launch_all_dashboards()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 