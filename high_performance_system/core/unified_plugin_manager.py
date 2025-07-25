"""Unified plugin system for LLM trust evaluation"""

import sys
import os
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class UnifiedTrustPlugin(ABC):
    """Base plugin interface for unified trust evaluation"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Plugin category (reliability, safety, fairness, etc.)"""
        pass
    
    @property
    @abstractmethod
    def model_types(self) -> List[str]:
        """Supported model types"""
        pass
    
    @abstractmethod
    def evaluate(self, model, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute evaluation"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if plugin dependencies are available"""
        pass

class UnifiedPluginManager:
    """Manages plugins across OpenTrustEval and TrustLLM"""
    
    def __init__(self):
        self.plugins: Dict[str, UnifiedTrustPlugin] = {}
        self.adapters: Dict[str, Any] = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize external tool adapters"""
        # TrustLLM adapter
        try:
            from src.integration.trustllm_adapter import TrustLLMAdapter
            self.adapters['trustllm'] = TrustLLMAdapter()
            logger.info("✓ TrustLLM adapter initialized")
        except Exception as e:
            logger.warning(f"⚠ TrustLLM adapter not available: {e}")
        
        # CleanLab adapter
        try:
            from src.integration.cleanlab_adapter import CleanLabAdapter
            self.adapters['cleanlab'] = CleanLabAdapter()
            logger.info("✓ CleanLab adapter initialized")
        except Exception as e:
            logger.warning(f"⚠ CleanLab adapter not available: {e}")
        
        # DeepChecks adapter
        try:
            from src.integration.deepchecks_adapter import DeepChecksAdapter
            self.adapters['deepchecks'] = DeepChecksAdapter()
            logger.info("✓ DeepChecks adapter initialized")
        except Exception as e:
            logger.warning(f"⚠ DeepChecks adapter not available: {e}")
    
    def register_plugin(self, plugin: UnifiedTrustPlugin):
        """Register a trust evaluation plugin"""
        if plugin.is_available():
            self.plugins[plugin.name] = plugin
            logger.info(f"✓ Registered plugin: {plugin.name}")
        else:
            logger.warning(f"⚠ Plugin not available: {plugin.name}")
    
    def get_compatible_plugins(self, model_type: str) -> List[UnifiedTrustPlugin]:
        """Get plugins compatible with specific model type"""
        compatible = []
        for plugin in self.plugins.values():
            if model_type in plugin.model_types or 'all' in plugin.model_types:
                compatible.append(plugin)
        return compatible
    
    def execute_evaluation(self, model, data: Dict[str, Any], 
                          model_type: str = 'llm', 
                          categories: List[str] = None) -> Dict[str, Any]:
        """Execute comprehensive evaluation using compatible plugins"""
        compatible_plugins = self.get_compatible_plugins(model_type)
        
        if categories:
            compatible_plugins = [p for p in compatible_plugins 
                                if p.category in categories]
        
        results = {}
        for plugin in compatible_plugins:
            try:
                plugin_result = plugin.evaluate(model, data)
                results[plugin.name] = {
                    'success': True,
                    'result': plugin_result,
                    'plugin_info': {
                        'name': plugin.name,
                        'category': plugin.category,
                        'model_types': plugin.model_types
                    }
                }
            except Exception as e:
                results[plugin.name] = {
                    'success': False,
                    'error': str(e),
                    'plugin_info': {
                        'name': plugin.name,
                        'category': plugin.category
                    }
                }
        
        return self._aggregate_results(results)
    
    def _aggregate_results(self, plugin_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple plugins"""
        aggregated = {
            'dimension_scores': {},
            'category_scores': {},
            'plugin_performance': {},
            'conflicts': [],
            'overall_score': 0.0
        }
        
        # Collect dimension scores
        for plugin_name, result in plugin_results.items():
            if result.get('success', False):
                plugin_data = result.get('result', {})
                if 'score' in plugin_data:
                    # Single score plugin
                    aggregated['plugin_performance'][plugin_name] = plugin_data['score']
                elif 'dimension_scores' in plugin_data:
                    # Multi-dimension plugin
                    for dim_name, dim_score in plugin_data['dimension_scores'].items():
                        if isinstance(dim_score, dict) and 'score' in dim_score:
                            aggregated['dimension_scores'][dim_name] = dim_score['score']
                        else:
                            aggregated['dimension_scores'][dim_name] = dim_score
        
        # Calculate category scores
        category_scores = {}
        for dim_name, score in aggregated['dimension_scores'].items():
            category = self._infer_category(dim_name)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
        
        for category, scores in category_scores.items():
            aggregated['category_scores'][category] = sum(scores) / len(scores)
        
        # Calculate overall score
        if aggregated['dimension_scores']:
            aggregated['overall_score'] = sum(aggregated['dimension_scores'].values()) / len(aggregated['dimension_scores'])
        elif aggregated['plugin_performance']:
            aggregated['overall_score'] = sum(aggregated['plugin_performance'].values()) / len(aggregated['plugin_performance'])
        
        return aggregated
    
    def _infer_category(self, dimension_name: str) -> str:
        """Infer category from dimension name"""
        dimension_name = dimension_name.lower()
        if any(word in dimension_name for word in ['truth', 'fact', 'accur']):
            return 'reliability'
        elif any(word in dimension_name for word in ['safe', 'harm', 'toxic']):
            return 'safety'
        elif any(word in dimension_name for word in ['fair', 'bias', 'discrim']):
            return 'fairness'
        elif any(word in dimension_name for word in ['consist', 'stable']):
            return 'consistency'
        elif any(word in dimension_name for word in ['explain', 'interpret']):
            return 'explainability'
        else:
            return 'general'

# Global plugin manager instance
plugin_manager = UnifiedPluginManager()
