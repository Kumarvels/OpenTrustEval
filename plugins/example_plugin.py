"""Plugin Development Template"""
# Example plugin: appends a note to the optimized decision

def custom_plugin(optimized_decision_dict):
    """
    Example plugin that appends a custom note to the optimized decision.
    Args:
        optimized_decision_dict (dict): {'optimized_decision': str}
    Returns:
        dict: {'plugin_output': str}
    """
    base = optimized_decision_dict['optimized_decision']
    return {'plugin_output': base + " [Plugin: Example note added]"}
