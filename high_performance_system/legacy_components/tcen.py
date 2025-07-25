"""Explainability Engine (TCEN) Module"""
# Implements Generic Explanation Module (GEM)

def explain_decision(decision_dict):
    """
    Generate an explanation for the decision.
    Args:
        decision_dict (dict): {'decision_score': float, 'decision_vector': np.ndarray}
    Returns:
        dict: {'explanation': str}
    """
    score = decision_dict['decision_score']
    # Simple placeholder explanation logic
    if score > 0:
        explanation = f"The decision score {score:.4f} indicates a positive trust assessment."
    elif score < 0:
        explanation = f"The decision score {score:.4f} indicates a negative trust assessment."
    else:
        explanation = "The decision score is neutral."
    return {'explanation': explanation}
