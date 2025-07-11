"""
API Endpoint: /thirdparty/verify_realtime
Verifies a single answer from a 3rd-party system in real time using OpenTrustEval.
"""
from fastapi import APIRouter, Form
from plugins.plugin_loader import load_plugins
from ote_api import process_pipeline

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
