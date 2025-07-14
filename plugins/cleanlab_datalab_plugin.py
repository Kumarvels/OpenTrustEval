"""
Cleanlab Datalab Plugin (Optional, Commercial/Consent Required)

This plugin integrates Cleanlab Datalab for automated data issue detection.
It is only activated if Cleanlab is installed and the plugin is enabled.

⚠️ Commercial/consent requirements: Cleanlab Datalab is a commercial offering. 
   You must have a valid license and user consent before using this plugin.
   See: https://cleanlab.ai/pricing/ and https://cleanlab.ai/docs/datalab/

Usage:
- Place this file in the plugins/ directory.
- Add 'cleanlab' to plugins/requirements.txt if you want to install it for this plugin only.
- Enable via config or plugin loader as needed.

Advanced Config Options:
- 'cleanlab_issue_types': list of issue types to include (e.g., ['label_error', 'outlier'])
- 'cleanlab_confidence_threshold': float, only include issues with confidence >= threshold
- 'cleanlab_datalab_kwargs': dict, extra kwargs to pass to Datalab constructor
"""

import importlib

# Try to import cleanlab, but do not fail if not present
try:
    cleanlab = importlib.import_module('cleanlab')
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

# Optional: Try to import pandas for DataFrame support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def cleanlab_datalab_plugin(data, labels=None, config=None):
    """
    Run Cleanlab Datalab on the provided data (if available and enabled).
    Args:
        data: List of dicts, DataFrame, or similar (text, label, etc.)
        labels: Optional list of labels
        config: Optional dict for plugin config (see docstring for options)
    Returns:
        dict: {'plugin_output': ..., 'issues': ..., 'warnings': ...}
    """
    warnings = []
    if not HAS_CLEANLAB:
        warnings.append("Cleanlab is not installed. Plugin is inactive.")
        return {'plugin_output': None, 'issues': None, 'warnings': warnings}
    if config is None or not config.get('enable_cleanlab', False):
        warnings.append("Cleanlab plugin is not enabled in config. Skipping analysis.")
        return {'plugin_output': None, 'issues': None, 'warnings': warnings}
    warnings.append("⚠️ Ensure you have a valid Cleanlab Datalab license and user consent before using this plugin.")
    # Convert data to DataFrame if possible
    if HAS_PANDAS and not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # Advanced config options
    issue_types = config.get('cleanlab_issue_types')
    confidence_threshold = config.get('cleanlab_confidence_threshold')
    datalab_kwargs = config.get('cleanlab_datalab_kwargs', {})
    # Minimal Cleanlab Datalab usage example
    try:
        from cleanlab.datalab import Datalab
        datalab = Datalab(data=data, labels=labels, **datalab_kwargs)
        datalab.find_issues()
        issues = datalab.get_issues()
        # Filter by issue type if specified
        if issue_types:
            issues = [iss for iss in issues if iss.get('issue_type') in issue_types]
        # Filter by confidence/score if specified
        if confidence_threshold is not None:
            issues = [iss for iss in issues if iss.get('confidence', 1.0) >= confidence_threshold]
        plugin_output = f"Detected {len(issues)} data issues via Cleanlab Datalab."
        return {'plugin_output': plugin_output, 'issues': issues, 'warnings': warnings}
    except Exception as e:
        warnings.append(f"Cleanlab Datalab error: {e}")
        return {'plugin_output': None, 'issues': None, 'warnings': warnings} 