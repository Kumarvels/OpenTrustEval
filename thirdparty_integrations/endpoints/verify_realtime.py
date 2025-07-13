"""
API Endpoint: /thirdparty/verify_realtime
Verifies a single answer from a 3rd-party system in real time using OpenTrustEval.
"""
from fastapi import APIRouter, Form
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
    print(f"✅ Verify Realtime integrated with high-performance system")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available for Verify Realtime: {e}")

def get_high_performance_integration_status():
    """Get high-performance integration status"""
    return {
        'available': HIGH_PERFORMANCE_AVAILABLE,
        'moe_system': 'active' if HIGH_PERFORMANCE_AVAILABLE and moe_system else 'inactive',
        'safety_layer': 'active' if HIGH_PERFORMANCE_AVAILABLE and safety_layer else 'inactive'
    }


router = APIRouter()

@router.post("/thirdparty/verify_realtime", summary="Verify a single answer in real time")
async def verify_realtime(
    answer: str = Form(...),
    image_url: str = Form(None),
    image_model: str = Form('EfficientNetB0'),
    min_trust: str = Form('Trustworthy')
):
    # TODO: Download image if image_url provided
    discovered_plugins = load_plugins('plugins')
    optimized, plugin_outputs, _ = process_pipeline(answer, None, image_model, discovered_plugins)
    halluc_flag = False
    for v in plugin_outputs.values():
        if isinstance(v, dict) and v.get('hallucination_flag'):
            halluc_flag = True
    trust_score = optimized['optimized_decision']
    allow = (trust_score == min_trust) and not halluc_flag
    return {
        'allow': allow,
        'trust_score': trust_score,
        'hallucination': halluc_flag,
        'plugin_outputs': plugin_outputs
    }
