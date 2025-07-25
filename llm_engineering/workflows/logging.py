import time

WORKFLOW_LOGS = []

def log_workflow_run(workflow_name, input_data, output, error=None):
    WORKFLOW_LOGS.append({
        "workflow": workflow_name,
        "input": input_data,
        "output": output,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

def get_workflow_logs():
    return WORKFLOW_LOGS 