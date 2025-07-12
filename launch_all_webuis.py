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

# Paths to scripts
DATASET_WEBUI = os.path.join('data_engineering', 'scripts', 'dataset_webui.py')
LLM_WEBUI = os.path.join('llm_engineering', 'scripts', 'llm_webui.py')
SECURITY_WEBUI = os.path.join('security', 'security_webui.py')

# Ports
DATASET_PORT = 7861
LLM_PORT = 7862
SECURITY_PORT = 7863

# Helper to launch a script in the background
processes = []
def launch_webui(script_path, port, name):
    if not os.path.exists(script_path):
        print(f"[ERROR] {name} script not found: {script_path}")
        return None
    try:
        proc = subprocess.Popen([
            sys.executable, script_path,
            f"--server_port={port}",
            "--server_name=0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[OK] Launched {name} at http://localhost:{port}")
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to launch {name}: {e}")
        return None

def open_browser(port, name):
    url = f"http://localhost:{port}"
    print(f"[INFO] Opening {name} in browser: {url}")
    webbrowser.open(url)

def main():
    print("\n=== OpenTrustEval One-Click WebUI Launcher ===\n")
    print("Launching all WebUIs in parallel...")
    print("- Dataset Management WebUI (port 7861)")
    print("- LLM Model Manager & Tuning WebUI (port 7862)")
    print("- Security Management WebUI (port 7863)")
    print("\nIf a UI fails to launch, check for missing dependencies (see README).\n")

    # Launch all WebUIs
    procs = []
    procs.append(launch_webui(DATASET_WEBUI, DATASET_PORT, "Dataset Management WebUI"))
    procs.append(launch_webui(LLM_WEBUI, LLM_PORT, "LLM Model Manager WebUI"))
    procs.append(launch_webui(SECURITY_WEBUI, SECURITY_PORT, "Security Management WebUI"))

    # Wait a bit for servers to start
    time.sleep(3)
    open_browser(DATASET_PORT, "Dataset Management WebUI")
    open_browser(LLM_PORT, "LLM Model Manager WebUI")
    open_browser(SECURITY_PORT, "Security Management WebUI")

    print("\nAll WebUIs launched. Press Ctrl+C to stop all.")
    try:
        # Wait for all processes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all WebUIs...")
        for p in procs:
            if p:
                p.terminate()
        print("All WebUIs stopped.")

if __name__ == "__main__":
    main() 