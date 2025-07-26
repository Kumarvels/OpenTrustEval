# src/correlation/cross_model_correlation.py
"""
Cross-Model Trust Correlation Engine - Understand how trust in one model affects others
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Any, List, Tuple
import networkx as nx
import plotly.graph_objects as go

class CrossModelCorrelationEngine:
    """Analyzes trust correlations between different AI models"""
    
    def __init__(self):
        self.model_network = nx.DiGraph()  # Directed graph for causality
        self.trust_correlations = {}
        self.interference_patterns = {}
    
    def register_model_relationship(self, source_model: str, target_model: str, 
                                  relationship_type: str, strength: float = 1.0):
        """Register a relationship between models"""
        self.model_network.add_edge(source_model, target_model, 
                                  relationship=relationship_type, 
                                  strength=strength)
    
    def analyze_trust_interference(self, model_evaluations: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how trust in one model affects trust in others"""
        interference_analysis = {}
        
        # Calculate correlation matrix
        model_names = list(model_evaluations.keys())
        trust_scores = {model: eval_data.get('overall_trust_score', 0.5) 
                       for model, eval_data in model_evaluations.items()}
        
        # Pairwise correlation analysis
        correlations = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    score1 = trust_scores[model1]
                    score2 = trust_scores[model2]
                    
                    # Simple correlation (in real implementation, use more sophisticated methods)
                    correlation = self._calculate_trust_correlation(
                        model_evaluations[model1], 
                        model_evaluations[model2]
                    )
                    
                    correlations[f"{model1}__{model2}"] = {
                        'correlation': correlation,
                        'trust_impact': self._calculate_trust_impact(score1, score2),
                        'risk_amplification': self._calculate_risk_amplification(
                            model_evaluations[model1], 
                            model_evaluations[model2]
                        )
                    }
        
        # Identify risk clusters
        risk_clusters = self._identify_risk_clusters(model_evaluations)
        
        # Calculate system-level trust
        system_trust = self._calculate_system_level_trust(model_evaluations)
        
        return {
            'pairwise_correlations': correlations,
            'risk_clusters': risk_clusters,
            'system_level_trust': system_trust,
            'vulnerability_analysis': self._analyze_vulnerability_propagation(model_evaluations),
            'recommendations': self._generate_correlation_recommendations(
                correlations, risk_clusters, system_trust
            )
        }
    
    def _calculate_trust_correlation(self, eval1: Dict, eval2: Dict) -> float:
        """Calculate trust correlation between two model evaluations"""
        # Extract dimension scores
        dims1 = eval1.get('dimension_scores', {})
        dims2 = eval2.get('dimension_scores', {})
        
        # Get common dimensions
        common_dims = set(dims1.keys()) & set(dims2.keys())
        
        if len(common_dims) < 2:
            return 0.0
        
        scores1 = [dims1[dim] for dim in common_dims]
        scores2 = [dims2[dim] for dim in common_dims]
        
        # Calculate Pearson correlation
        try:
            correlation, _ = pearsonr(scores1, scores2)
            return float(correlation)
        except:
            return 0.0
    
    def _calculate_trust_impact(self, source_trust: float, target_trust: float) -> Dict[str, float]:
        """Calculate how source model trust impacts target model trust"""
        # Simple impact model - in practice, this would be more sophisticated
        impact_magnitude = abs(source_trust - 0.5) * abs(target_trust - 0.5) * 2
        impact_direction = 1 if (source_trust > 0.5 and target_trust > 0.5) or \
                              (source_trust < 0.5 and target_trust < 0.5) else -1
        
        return {
            'magnitude': impact_magnitude,
            'direction': impact_direction,  # 1 = positive correlation, -1 = negative
            'risk_factor': impact_magnitude * abs(impact_direction)
        }
    
    def _calculate_risk_amplification(self, eval1: Dict, eval2: Dict) -> float:
        """Calculate how risks in one model amplify risks in another"""
        # Extract risk scores
        risks1 = eval1.get('risk_assessment', {}).get('high_risks', [])
        risks2 = eval2.get('risk_assessment', {}).get('high_risks', [])
        
        # Simple risk amplification model
        base_risk = len(risks1) + len(risks2)
        amplified_risk = base_risk * (1 + len(risks1) * len(risks2) * 0.1)
        
        return min(10.0, amplified_risk)  # Cap at reasonable level
    
    def _identify_risk_clusters(self, model_evaluations: Dict[str, Dict]) -> List[Dict]:
        """Identify clusters of models with correlated risks"""
        # Simple clustering based on correlation thresholds
        clusters = []
        processed_models = set()
        
        for model_name, eval_data in model_evaluations.items():
            if model_name in processed_models:
                continue
            
            # Find correlated models
            correlated_models = []
            for other_model, other_eval in model_evaluations.items():
                if other_model != model_name and other_model not in processed_models:
                    correlation = self._calculate_trust_correlation(eval_data, other_eval)
                    if abs(correlation) > 0.7:  # High correlation threshold
                        correlated_models.append(other_model)
                        processed_models.add(other_model)
            
            if correlated_models:
                correlated_models.append(model_name)
                clusters.append({
                    'models': correlated_models,
                    'cluster_risk': self._calculate_cluster_risk(
                        {m: model_evaluations[m] for m in correlated_models}
                    )
                })
                processed_models.add(model_name)
        
        return clusters
    
    def _calculate_system_level_trust(self, model_evaluations: Dict[str, Dict]) -> float:
        """Calculate overall system trust considering correlations"""
        if not model_evaluations:
            return 0.5
        
        individual_trusts = [eval_data.get('overall_trust_score', 0.5) 
                           for eval_data in model_evaluations.values()]
        
        # Simple average - in practice, weight by model importance and correlations
        return float(np.mean(individual_trusts))
    
    def _analyze_vulnerability_propagation(self, model_evaluations: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how vulnerabilities might propagate through the system"""
        propagation_analysis = {}
        
        for model_name, eval_data in model_evaluations.items():
            vulnerabilities = eval_data.get('risk_assessment', {}).get('critical_risks', [])
            
            if vulnerabilities:
                propagation_analysis[model_name] = {
                    'vulnerabilities': vulnerabilities,
                    'propagation_risk': len(vulnerabilities) * 0.2,
                    'affected_downstream': self._find_affected_models(model_name),
                    'mitigation_priority': self._calculate_mitigation_priority(eval_data)
                }
        
        return propagation_analysis
    
    def _find_affected_models(self, source_model: str) -> List[str]:
        """Find models that might be affected by issues in source model"""
        if source_model in self.model_network:
            return list(self.model_network.successors(source_model))
        return []
    
    def _calculate_mitigation_priority(self, eval_data: Dict) -> str:
        """Calculate priority for mitigation based on risk severity"""
        critical_risks = len(eval_data.get('risk_assessment', {}).get('critical_risks', []))
        high_risks = len(eval_data.get('risk_assessment', {}).get('high_risks', []))
        
        risk_score = critical_risks * 3 + high_risks
        
        if risk_score >= 6:
            return 'critical'
        elif risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_correlation_recommendations(self, correlations: Dict, 
                                            clusters: List[Dict], 
                                            system_trust: float) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        
        # System-level recommendations
        if system_trust < 0.6:
            recommendations.append("System-level trust is low. Consider comprehensive system review.")
        
        # Cluster recommendations
        high_risk_clusters = [c for c in clusters if c['cluster_risk'] > 0.7]
        if high_risk_clusters:
            recommendations.append(f"Identified {len(high_risk_clusters)} high-risk model clusters. Review interdependencies.")
        
        # Correlation recommendations
        strong_correlations = [corr for corr, data in correlations.items() 
                              if abs(data['correlation']) > 0.8]
        if strong_correlations:
            recommendations.append(f"Found {len(strong_correlations)} strong model correlations. Monitor jointly.")
        
        return recommendations

# Integration example
class CorrelationAwareEvaluator:
    """Evaluator that considers cross-model correlations"""
    
    def __init__(self):
        self.correlation_engine = CrossModelCorrelationEngine()
        # ... other initialization
    
    def evaluate_multi_model_system(self, models: Dict[str, Any], 
                                  shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a system with multiple interacting models"""
        
        # Evaluate each model individually
        individual_evaluations = {}
        for model_name, model in models.items():
            individual_evaluations[model_name] = self.evaluate_comprehensive_trust(
                model, shared_data.get(model_name, shared_data)
            )
        
        # Analyze cross-model correlations
        correlation_analysis = self.correlation_engine.analyze_trust_interference(
            individual_evaluations
        )
        
        return {
            'individual_evaluations': individual_evaluations,
            'correlation_analysis': correlation_analysis,
            'system_overview': {
                'total_models': len(models),
                'system_trust': correlation_analysis['system_level_trust'],
                'risk_clusters': len(correlation_analysis['risk_clusters'])
            }
        }
