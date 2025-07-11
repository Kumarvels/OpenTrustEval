"""Advanced Plugin Template: Hallucination Detector"""
# This plugin flags possible hallucinations in the decision output.

def hallucination_detector_plugin(optimized_decision_dict):
    """
    Flags possible hallucinations in the optimized decision output.
    Args:
        optimized_decision_dict (dict): {'optimized_decision': str}
    Returns:
        dict: {'plugin_output': str, 'hallucination_flag': bool}
    """
    output = optimized_decision_dict['optimized_decision']
    # Example logic: flag if certain keywords are present (placeholder)
    hallucination_keywords = ["unreal", "impossible", "fabricated", "hallucination"]
    flag = any(word in output.lower() for word in hallucination_keywords)
    note = " [Hallucination Detected]" if flag else " [No Hallucination]"
    return {
        'plugin_output': output + note,
        'hallucination_flag': flag
    }
