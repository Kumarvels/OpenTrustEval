import yaml
import json
import os

def load_workflow_template(path):
    ext = os.path.splitext(path)[-1].lower()
    with open(path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError("Unsupported template format: " + ext) 