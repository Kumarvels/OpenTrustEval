#!/usr/bin/env python3
"""
One-Click Launcher for OpenTrustEval WebUIs

This script launches all major WebUIs in parallel:
- Dataset Management WebUI (Gradio, port 7861)
- LLM Model Manager & Tuning WebUI (Gradio, port 7862)
- Security Management WebUI (Gradio, port 7863)

Usage:
    python launch_all_webuis.py

Each UI will open in your browser automatically. Press Ctrl+C to stop all.
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse

# Paths to scripts
DATASET_WEBUI = os.path.join('data_engineering', 'scripts', 'easy_dataset_webui.py')
LLM_WEBUI = os.path.join('llm_engineering', 'scripts', 'llm_webui.py')
SECURITY_WEBUI = os.path.join('security', 'security_webui.py')

def check_script_exists(script_path, name):
    """Check if a script exists and is executable"""
    if not os.path.exists(script_path):
        print(f"❌ {name} script not found: {script_path}")
        return False
    return True

def launch_webui(script_path, port, name):
    """Launch a WebUI in a subprocess"""
    try:
        print(f"🚀 Starting {name} on port {port}...")
        
        # Use python executable from current environment
        python_exe = sys.executable
        
        # Launch the WebUI
        process = subprocess.Popen(
            [python_exe, script_path, '--server_port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment to see if it starts successfully
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ {name} started successfully on port {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ {name} failed to start:")
            if stdout:
                print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching {name}: {e}")
        return None

def open_browser(port, name):
    """Open browser to WebUI"""
    try:
        url = f"http://localhost:{port}"
        print(f"🌐 Opening {name} in browser: {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"⚠️ Could not open browser for {name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Launch all OpenTrustEval WebUIs")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browsers automatically")
    parser.add_argument("--dataset-port", type=int, default=7861, help="Dataset WebUI port")
    parser.add_argument("--llm-port", type=int, default=7862, help="LLM WebUI port")
    parser.add_argument("--security-port", type=int, default=7863, help="Security WebUI port")
    args = parser.parse_args()
    
    print("🎯 OpenTrustEval WebUI Launcher")
    print("=" * 50)
    
    # Check if scripts exist
    scripts_to_launch = []
    
    if check_script_exists(DATASET_WEBUI, "Dataset"):
        scripts_to_launch.append(("Dataset Management", DATASET_WEBUI, args.dataset_port))
    
    if check_script_exists(LLM_WEBUI, "LLM"):
        scripts_to_launch.append(("LLM Model Manager", LLM_WEBUI, args.llm_port))
    
    if check_script_exists(SECURITY_WEBUI, "Security"):
        scripts_to_launch.append(("Security Management", SECURITY_WEBUI, args.security_port))
    
    if not scripts_to_launch:
        print("❌ No WebUI scripts found. Please check the file paths.")
        return
    
    print(f"\n📋 Found {len(scripts_to_launch)} WebUI(s) to launch:")
    for name, script, port in scripts_to_launch:
        print(f"  - {name}: {script} (port {port})")
    
    print("\n🚀 Launching WebUIs...")
    
    # Launch all WebUIs
    processes = []
    for name, script, port in scripts_to_launch:
        process = launch_webui(script, port, name)
        if process:
            processes.append((name, process, port))
        else:
            print(f"⚠️ Skipping {name} due to launch failure")
    
    if not processes:
        print("❌ No WebUIs were successfully launched.")
        return
    
    print(f"\n✅ Successfully launched {len(processes)} WebUI(s)")
    
    # Open browsers
    if not args.no_browser:
        print("\n🌐 Opening WebUIs in browser...")
        for name, process, port in processes:
            open_browser(port, name)
            time.sleep(1)  # Small delay between browser opens
    
    print("\n📊 WebUI Status:")
    for name, process, port in processes:
        status = "🟢 Running" if process.poll() is None else "🔴 Stopped"
        print(f"  - {name} (port {port}): {status}")
    
    print(f"\n🔗 WebUI URLs:")
    for name, process, port in processes:
        print(f"  - {name}: http://localhost:{port}")
    
    print("\n⏹️ Press Ctrl+C to stop all WebUIs...")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            # Check if any process has stopped
            for name, process, port in processes[:]:
                if process.poll() is not None:
                    print(f"⚠️ {name} has stopped unexpectedly")
                    processes.remove((name, process, port))
            
            if not processes:
                print("❌ All WebUIs have stopped.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping all WebUIs...")
        
        # Terminate all processes
        for name, process, port in processes:
            try:
                process.terminate()
                print(f"🛑 Stopped {name}")
            except Exception as e:
                print(f"⚠️ Error stopping {name}: {e}")
        
        # Wait for processes to terminate
        time.sleep(2)
        
        # Force kill if still running
        for name, process, port in processes:
            if process.poll() is None:
                try:
                    process.kill()
                    print(f"💀 Force killed {name}")
                except Exception as e:
                    print(f"⚠️ Error force killing {name}: {e}")
        
        print("✅ All WebUIs stopped.")

if __name__ == "__main__":
    main() 