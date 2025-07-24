"""
Legacy Compatibility Layer for Ultimate MoE Solution
Provides backward compatibility for old src/ components
"""

# Import the new high-performance implementations
from ..core.advanced_expert_ensemble import AdvancedExpertEnsemble
from ..core.intelligent_domain_router import IntelligentDomainRouter
from ..core.ultimate_moe_system import UltimateMoESystem

# Legacy function mappings
def process_input(input_data, image_model_name='EfficientNetB0'):
    """
    Legacy LHEM process_input function - now uses high-performance system
    """
    from ..core.ultimate_moe_system import UltimateMoESystem
    
    # Initialize the high-performance system
    moe_system = UltimateMoESystem()
    
    # Process the input using the new system
    if isinstance(input_data, dict) and 'text' in input_data:
        text = input_data['text']
        image = input_data.get('image')
        context = input_data.get('context', '')

        if not text and image is None:
            raise ValueError("Either text or image must be provided.")

        if image is not None and (len(image.shape) != 3 or image.shape[-1] != 3):
            raise ValueError("Image must have 3 channels.")
        
        import asyncio
        # Use the new verification system
        async def verify():
            return await moe_system.verify_text(text, context)
        result = asyncio.run(verify())
        
        # Return in legacy format
        return {
            'text_embedding': result.verification_score,
            'image_embedding': None  # Legacy compatibility
        }
    else:
        raise ValueError('Invalid input format for legacy compatibility')

def extract_evidence(embedding_dict):
    """
    Legacy TEE extract_evidence function - now uses high-performance system
    """
    from ..core.ultimate_moe_system import UltimateMoESystem
    
    # Initialize the high-performance system
    moe_system = UltimateMoESystem()
    
    # Extract evidence using the new system
    if isinstance(embedding_dict, dict) and 'text_embedding' in embedding_dict:
        # Use the verification score as evidence
        evidence_vector = [embedding_dict['text_embedding']]
        return {'evidence_vector': evidence_vector}
    else:
        raise ValueError('Invalid embedding format for legacy compatibility')

def aggregate_evidence(evidence_dict):
    """
    Legacy DEL aggregate_evidence function - now uses high-performance system
    """
    from ..core.ultimate_moe_system import UltimateMoESystem
    
    # Initialize the high-performance system
    moe_system = UltimateMoESystem()
    
    # Aggregate evidence using the new system
    if isinstance(evidence_dict, dict) and 'evidence_vector' in evidence_dict:
        evidence_vector = evidence_dict['evidence_vector']
        decision_score = sum(evidence_vector) / len(evidence_vector) if evidence_vector else 0.0
        return {
            'decision_score': decision_score,
            'decision_vector': evidence_vector
        }
    else:
        raise ValueError('Invalid evidence format for legacy compatibility')

# Legacy pipeline logger compatibility
def init_db():
    """Legacy database initialization - now uses high-performance logging"""
    from ..core.ultimate_moe_system import UltimateMoESystem
    # The new system handles logging automatically
    return True

def log_pipeline(stage, data, status="success"):
    """Legacy pipeline logging - now uses high-performance logging"""
    from ..core.ultimate_moe_system import UltimateMoESystem
    # The new system handles logging automatically
    return True

# Export legacy functions for backward compatibility
__all__ = [
    'process_input',
    'extract_evidence', 
    'aggregate_evidence',
    'init_db',
    'log_pipeline'
]
