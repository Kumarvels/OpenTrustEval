"""TrustLLM comprehensive trust evaluation plugin"""

from src.core.unified_plugin_manager import UnifiedTrustPlugin
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TrustLLMComprehensivePlugin(UnifiedTrustPlugin):
    """Comprehensive TrustLLM evaluation plugin"""
    
    def __init__(self):
        self.name = "trustllm_comprehensive"
        self.category = "overall"
        self.model_types = ["llm", "language_model", "transformer"]
        self.adapter = None
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize TrustLLM adapter"""
        try:
            from src.integration.trustllm_adapter import TrustLLMAdapter
            self.adapter = TrustLLMAdapter()
        except Exception as e:
            logger.error(f"Failed to initialize TrustLLM adapter: {e}")
            self.adapter = None
    
    def is_available(self) -> bool:
        """Check if plugin is available"""
        return self.adapter is not None and self.adapter.is_available()
    
    def evaluate(self, model, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute comprehensive TrustLLM evaluation"""
        if not self.is_available():
            return self._fallback_evaluation()
        
        try:
            # Prepare data for TrustLLM
            trustllm_data = self._prepare_trustllm_data(data)
            
            # Execute TrustLLM evaluations
            results = {}
            
            # Truthfulness evaluation
            truthfulness_result = self.adapter.evaluate_truthfulness(model, trustllm_data)
            results['truthfulness'] = truthfulness_result
            
            # Safety evaluation
            safety_result = self.adapter.evaluate_safety(model, trustllm_data)
            results['safety'] = safety_result
            
            # Hallucination evaluation
            hallucination_result = self.adapter.evaluate_hallucination(model, trustllm_data)
            results['hallucination'] = hallucination_result
            
            # Privacy evaluation
            privacy_result = self.adapter.evaluate_privacy(model, trustllm_data)
            results['privacy'] = privacy_result
            
            # Toxicity evaluation
            toxicity_result = self.adapter.evaluate_toxicity(model, trustllm_data)
            results['toxicity'] = toxicity_result
            
            # Calculate comprehensive score
            dimension_scores = {}
            for dim_name, dim_result in results.items():
                if isinstance(dim_result, dict) and 'score' in dim_result:
                    dimension_scores[dim_name] = dim_result['score']
            
            comprehensive_score = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.5
            
            return {
                'score': float(comprehensive_score),
                'dimension_scores': dimension_scores,
                'detailed_results': results,
                'metadata': {
                    'evaluator': 'trustllm',
                    'timestamp': self._get_timestamp(),
                    'dimensions_evaluated': list(dimension_scores.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"TrustLLM evaluation failed: {e}")
            return self._fallback_evaluation()
    
    def _prepare_trustllm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data in TrustLLM format"""
        # Convert OpenTrustEval data format to TrustLLM format
        trustllm_data = {}
        
        # Map common fields
        if 'prompts' in data:
            trustllm_data['prompts'] = data['prompts']
        if 'responses' in data:
            trustllm_data['responses'] = data['responses']
        if 'ground_truth' in data:
            trustllm_data['ground_truth'] = data['ground_truth']
        if 'contexts' in data:
            trustllm_data['contexts'] = data['contexts']
        
        return trustllm_data
    
    def _fallback_evaluation(self) -> Dict[str, Any]:
        """Fallback evaluation when TrustLLM unavailable"""
        return {
            'score': 0.5,
            'dimension_scores': {
                'truthfulness': 0.5,
                'safety': 0.7,  # Conservative safety default
                'hallucination': 0.5,
                'privacy': 0.6,
                'toxicity': 0.7
            },
            'detailed_results': {},
            'metadata': {
                'evaluator': 'fallback',
                'warning': 'TrustLLM not available, using default scores'
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Register the plugin
def register_trustllm_plugin(plugin_manager):
    """Register TrustLLM plugin with plugin manager"""
    plugin = TrustLLMComprehensivePlugin()
    if plugin.is_available():
        plugin_manager.register_plugin(plugin)
        logger.info("✓ TrustLLM comprehensive plugin registered")
    else:
        logger.warning("⚠ TrustLLM comprehensive plugin not available")

# Auto-registration
try:
    register_trustllm_plugin(plugin_manager)
except Exception as e:
    logger.error(f"Failed to register TrustLLM plugin: {e}")
