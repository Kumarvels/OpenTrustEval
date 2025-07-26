import subprocess
import sys

def main():
    """
    Launches the Unified Workflow Web UI.
    """
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "workflow_webui.py"])
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Please make sure Streamlit is installed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
