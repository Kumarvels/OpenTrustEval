"""LLM-specific reliability evaluation plugin"""

from src.core.unified_plugin_manager import UnifiedTrustPlugin
from typing import Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LLMReliabilityPlugin(UnifiedTrustPlugin):
    """LLM-specific reliability evaluation plugin"""
    
    def __init__(self):
        self.name = "llm_reliability"
        self.category = "reliability"
        self.model_types = ["llm", "language_model", "transformer", "all"]
        self.adapter = None
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize adapter for external tools"""
        try:
            from src.integration.cleanlab_adapter import CleanLabAdapter
            self.adapter = CleanLabAdapter()
        except Exception as e:
            logger.debug(f"CleanLab adapter not available: {e}")
            self.adapter = None
    
    def is_available(self) -> bool:
        """Plugin is always available (fallback methods)"""
        return True
    
    def evaluate(self, model, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute LLM reliability evaluation"""
        try:
            # Extract prompts and generate responses
            prompts = data.get('prompts', [])
            if not prompts:
                return self._basic_reliability_evaluation()
            
            # Generate responses
            responses = []
            for prompt in prompts:
                response = self._generate_response(model, prompt)
                responses.append(response)
            
            # Evaluate different aspects of reliability
            factual_accuracy = self._evaluate_factual_accuracy(data, responses)
            consistency_score = self._evaluate_consistency(model, prompts, responses)
            confidence_calibration = self._evaluate_confidence_calibration(model, prompts, responses)
            
            # Calculate overall reliability score
            reliability_components = [
                factual_accuracy,
                consistency_score,
                confidence_calibration
            ]
            overall_reliability = sum(reliability_components) / len(reliability_components)
            
            return {
                'score': float(overall_reliability),
                'dimension_scores': {
                    'factual_accuracy': factual_accuracy,
                    'consistency': consistency_score,
                    'confidence_calibration': confidence_calibration
                },
                'details': {
                    'evaluated_prompts': len(prompts),
                    'response_consistency': self._analyze_response_patterns(responses)
                },
                'metadata': {
                    'evaluator': 'llm_reliability_plugin',
                    'timestamp': self._get_timestamp()
                }
            }
            
        except Exception as e:
            logger.error(f"LLM reliability evaluation failed: {e}")
            return self._basic_reliability_evaluation()
    
    def _generate_response(self, model, prompt: str) -> str:
        """Generate response from model"""
        try:
            if hasattr(model, 'generate'):
                return model.generate(prompt)
            elif
