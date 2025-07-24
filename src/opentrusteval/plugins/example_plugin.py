"""Plugin Development Template"""
# Example plugin: appends a note to the optimized decision

def custom_plugin(optimized_decision_dict):
    """
    Example plugin that appends a custom note to the optimized decision.
    Args:
        optimized_decision_dict (dict): {'decision_score': float, 'decision_vector': list}
    Returns:
        dict: {'plugin_output': str}
    """
    base = optimized_decision_dict['decision_score']
    return {'plugin_output': str(base) + " [Plugin: Example note added]"}
