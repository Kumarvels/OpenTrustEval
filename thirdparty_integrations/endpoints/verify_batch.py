"""
API Endpoint: /thirdparty/verify_batch
Verifies a batch of answers from a 3rd-party system using OpenTrustEval.
"""
from fastapi import APIRouter, Body
from plugins.plugin_loader import load_plugins
from ote_api import process_pipeline
# High-Performance System Integration
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
    
    # Initialize high-performance components
    moe_system = UltimateMoESystem()
    expert_ensemble = AdvancedExpertEnsemble()
    
    HIGH_PERFORMANCE_AVAILABLE = True
    print(f"✅ Verify Batch integrated with high-performance system")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available for Verify Batch: {e}")

def get_high_performance_integration_status():
    """Get high-performance integration status"""
    return {
        'available': HIGH_PERFORMANCE_AVAILABLE,
        'moe_system': 'active' if HIGH_PERFORMANCE_AVAILABLE and moe_system else 'inactive',
        'safety_layer': 'active' if HIGH_PERFORMANCE_AVAILABLE and safety_layer else 'inactive'
    }


router = APIRouter()

@router.post("/thirdparty/verify_batch", summary="Verify a batch of answers")
async def verify_batch(
    items: list = Body(..., example=[{"answer": "text", "image_url": None, "image_model": "EfficientNetB0"}]),
    min_trust: str = 'Trustworthy'
):
    discovered_plugins = load_plugins('plugins')
    results = []
    for item in items:
        answer = item.get('answer')
        # TODO: Download image if image_url provided
        optimized, plugin_outputs, _ = process_pipeline(answer, None, item.get('image_model', 'EfficientNetB0'), discovered_plugins)
        halluc_flag = False
        for v in plugin_outputs.values():
            if isinstance(v, dict) and v.get('hallucination_flag'):
                halluc_flag = True
        trust_score = optimized['optimized_decision']
        allow = (trust_score == min_trust) and not halluc_flag
        results.append({
            'allow': allow,
            'trust_score': trust_score,
            'hallucination': halluc_flag,
            'plugin_outputs': plugin_outputs
        })
    return {'results': results}
