# src/decision/trust_decision_matrix.py
"""
Trust Decision Matrix - Customizable trust criteria for different stakeholders and contexts
"""

from typing import Dict, Any, List, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

class StakeholderType(Enum):
    """Different types of stakeholders with varying trust requirements"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    END_USER = "end_user"
    BUSINESS = "business"

class DeploymentContext(Enum):
    """Different deployment contexts with varying risk tolerances"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CRITICAL = "critical"

@dataclass
class TrustCriterion:
    """A specific trust criterion with threshold and weight"""
    dimension: str
    threshold: float
    weight: float
    critical: bool = False
    rationale: str = ""

class TrustDecisionMatrix:
    """Matrix-based trust decision system"""
    
    def __init__(self):
        self.decision_profiles = {}
        self.default_profiles = self._create_default_profiles()
    
    def _create_default_profiles(self) -> Dict[str, List[TrustCriterion]]:
        """Create default decision profiles for common scenarios"""
        return {
            'executive_high_level': [
                TrustCriterion('overall_trust', 0.8, 1.0, rationale="Executive overview"),
                TrustCriterion('safety', 0.9, 0.8, critical=True, rationale="Safety is paramount"),
                TrustCriterion('reliability', 0.85, 0.7, rationale="Business reliability")
            ],
            'technical_detailed': [
                TrustCriterion('reliability', 0.8, 0.9, rationale="Technical accuracy"),
                TrustCriterion('consistency', 0.85, 0.8, rationale="Stable performance"),
                TrustCriterion('robustness', 0.8, 0.7, rationale="Resilience to attacks"),
                TrustCriterion('explainability', 0.7, 0.6, rationale="Debugging capability")
            ],
            'regulatory_compliance': [
                TrustCriterion('fairness', 0.9, 1.0, critical=True, rationale="Non-discrimination"),
                TrustCriterion('privacy', 0.95, 0.9, critical=True, rationale="Data protection"),
                TrustCriterion('safety', 0.9, 0.8, critical=True, rationale="User safety"),
                TrustCriterion('transparency', 0.8, 0.7, rationale="Audit requirements")
            ],
            'production_critical': [
                TrustCriterion('safety', 0.95, 1.0, critical=True, rationale="Life-critical systems"),
                TrustCriterion('reliability', 0.9, 0.9, critical=True, rationale="Mission-critical"),
                TrustCriterion('robustness', 0.85, 0.8, rationale="Security resilience")
            ]
        }
    
    def create_custom_profile(self, profile_name: str, criteria: List[TrustCriterion]):
        """Create a custom decision profile"""
        self.decision_profiles[profile_name] = criteria
    
    def evaluate_against_profile(self, evaluation_results: Dict[str, Any], 
                               profile_name: str) -> Dict[str, Any]:
        """Evaluate trust results against a specific decision profile"""
        # Get profile criteria
        if profile_name in self.decision_profiles:
            criteria = self.decision_profiles[profile_name]
        elif profile_name in self.default_profiles:
            criteria = self.default_profiles[profile_name]
        else:
            return {'error': f'Profile {profile_name} not found'}
        
        # Extract dimension scores
        dimension_scores = evaluation_results.get('dimension_scores', {})
        category_scores = evaluation_results.get('category_scores', {})
        
        # Evaluate each criterion
        criterion_results = []
        critical_failures = []
        weighted_scores = []
        
        for criterion in criteria:
            # Get score for dimension
            score = dimension_scores.get(criterion.dimension)
            if score is None:
                # Try category score
                score = category_scores.get(criterion.dimension, 0.5)
            
            # Check threshold
            meets_threshold = score >= criterion.threshold
            if criterion.critical and not meets_threshold:
                critical_failures.append(criterion.dimension)
            
            # Calculate weighted contribution
            weighted_score = score * criterion.weight
            weighted_scores.append(weighted_score)
            
            criterion_results.append({
                'dimension': criterion.dimension,
                'score': score,
                'threshold': criterion.threshold,
                'meets_threshold': meets_threshold,
                'weight': criterion.weight,
                'weighted_contribution': weighted_score,
                'critical': criterion.critical
            })
        
        # Calculate overall profile score
        overall_profile_score = sum(weighted_scores) / sum(criterion.weight for criterion in criteria) if criteria else 0.5
        
        # Make decision
        decision = "APPROVED" if len(critical_failures) == 0 and overall_profile_score >= 0.7 else "REJECTED"
        if len(critical_failures) > 0:
            decision = "REJECTED_CRITICAL_FAILURES"
        
        return {
            'profile_name': profile_name,
            'overall_score': overall_profile_score,
            'decision': decision,
            'criterion_results': criterion_results,
            'critical_failures': critical_failures,
            'met_thresholds': len([c for c in criterion_results if c['meets_threshold']]),
            'total_criteria': len(criteria),
            'recommendations': self._generate_profile_recommendations(criterion_results)
        }
    
    def multi_profile_evaluation(self, evaluation_results: Dict[str, Any], 
                               profile_names: List[str]) -> Dict[str, Any]:
        """Evaluate against multiple profiles simultaneously"""
        profile_results = {}
        for profile_name in profile_names:
            profile_results[profile_name] = self.evaluate_against_profile(
                evaluation_results, profile_name
            )
        
        # Aggregate decisions
        final_decision = self._aggregate_decisions(profile_results)
        
        return {
            'individual_profile_results': profile_results,
            'final_decision': final_decision,
            'consensus_score': self._calculate_consensus_score(profile_results),
            'conflicting_decisions': self._find_conflicting_decisions(profile_results)
        }
    
    def _generate_profile_recommendations(self, criterion_results: List[Dict]) -> List[str]:
        """Generate recommendations based on profile evaluation"""
        recommendations = []
        
        for criterion in criterion_results:
            if not criterion['meets_threshold']:
                if criterion['critical']:
                    recommendations.append(f"CRITICAL: Improve {criterion['dimension']} (current: {criterion['score']:.3f}, required: {criterion['threshold']})")
                else:
                    recommendations.append(f"Improve {criterion['dimension']} (current: {criterion['score']:.3f}, required: {criterion['threshold']})")
        
        return recommendations
    
    def _aggregate_decisions(self, profile_results: Dict[str, Dict]) -> str:
        """Aggregate decisions from multiple profiles"""
        decisions = [result['decision'] for result in profile_results.values()]
        
        if 'REJECTED_CRITICAL_FAILURES' in decisions:
            return 'REJECTED_CRITICAL_FAILURES'
        elif 'REJECTED' in decisions:
            return 'REJECTED'
        else:
            return 'APPROVED'
    
    def _calculate_consensus_score(self, profile_results: Dict[str, Dict]) -> float:
        """Calculate consensus score across profiles"""
        scores = [result['overall_score'] for result in profile_results.values()]
        return float(np.mean(scores)) if scores else 0.5
    
    def _find_conflicting_decisions(self, profile_results: Dict[str, Dict]) -> List[str]:
        """Find profiles with conflicting decisions"""
        approved_profiles = [name for name, result in profile_results.items() 
                           if result['decision'] == 'APPROVED']
        rejected_profiles = [name for name, result in profile_results.items() 
                           if result['decision'] in ['REJECTED', 'REJECTED_CRITICAL_FAILURES']]
        
        if approved_profiles and rejected_profiles:
            return [f"Approved: {approved_profiles}, Rejected: {rejected_profiles}"]
        return []

# Integration with main system
class DecisionMatrixEvaluator:
    """Evaluator with decision matrix capabilities"""
    
    def __init__(self):
        self.decision_matrix = TrustDecisionMatrix()
        # ... other initialization
    
    def evaluate_with_decision_matrix(self, model, data, 
                                    stakeholder_type: StakeholderType = None,
                                    deployment_context: DeploymentContext = None,
                                    custom_profiles: List[str] = None) -> Dict[str, Any]:
        """Execute evaluation with decision matrix analysis"""
        
        # Standard evaluation
        evaluation_results = self.evaluate_comprehensive_trust(model, data)
        
        # Determine profiles to evaluate against
        profiles_to_evaluate = []
        
        if custom_profiles:
            profiles_to_evaluate.extend(custom_profiles)
        elif stakeholder_type:
            profile_mapping = {
                StakeholderType.EXECUTIVE: ['executive_high_level'],
                StakeholderType.TECHNICAL: ['technical_detailed'],
                StakeholderType.REGULATORY: ['regulatory_compliance'],
                StakeholderType.END_USER: ['executive_high_level'],
                StakeholderType.BUSINESS: ['executive_high_level']
            }
            profiles_to_evaluate.extend(profile_mapping.get(stakeholder_type, []))
        
        if deployment_context == DeploymentContext.CRITICAL:
            profiles_to_evaluate.append('production_critical')
        elif deployment_context == DeploymentContext.PRODUCTION:
            profiles_to_evaluate.append('technical_detailed')
        
        # If no specific profiles, use default comprehensive evaluation
        if not profiles_to_evaluate:
            profiles_to_evaluate = ['executive_high_level', 'technical_detailed']
        
        # Multi-profile evaluation
        decision_results = self.decision_matrix.multi_profile_evaluation(
            evaluation_results, profiles_to_evaluate
        )
        
        # Combine results
        final_results = evaluation_results.copy()
        final_results['decision_matrix_analysis'] = decision_results
        
        return final_results

# Usage example
def advanced_trust_decision_example():
    """Example of advanced trust decision making"""
    
    # Create evaluator
    evaluator = DecisionMatrixEvaluator()
    
    # Define custom profile for healthcare application
    healthcare_criteria = [
        TrustCriterion('safety', 0.95, 1.0, critical=True, rationale="Patient safety"),
        TrustCriterion('privacy', 0.95, 0.9, critical=True, rationale="HIPAA compliance"),
        TrustCriterion('reliability', 0.9, 0.8, rationale="Medical accuracy"),
        TrustCriterion('fairness', 0.9, 0.7, rationale="Non-discrimination")
    ]
    
    evaluator.decision_matrix.create_custom_profile('healthcare_medical_ai', healthcare_criteria)
    
    # Evaluate model
    results = evaluator.evaluate_with_decision_matrix(
        model=my_medical_ai_model,
        data=medical_test_data,
        custom_profiles=['healthcare_medical_ai', 'regulatory_compliance']
    )
    
    print(f"Overall Trust Score: {results['overall_trust_score']:.3f}")
    print(f"Decision: {results['decision_matrix_analysis']['final_decision']}")
    
    # Show detailed analysis
    for profile_name, profile_result in results['decision_matrix_analysis']['individual_profile_results'].items():
        print(f"\n{profile_name.upper()} Profile:")
        print(f"  Score: {profile_result['overall_score']:.3f}")
        print(f"  Decision: {profile_result['decision']}")
        if profile_result['recommendations']:
            print("  Recommendations:")
            for rec in profile_result['recommendations']:
                print(f"    - {rec}")
