# src/simulation/trust_simulation.py
"""
Trust Simulation and Stress Testing - Test trust under extreme conditions
"""

import numpy as np
from typing import Dict, Any, List, Callable
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SimulationScenario:
    """A specific simulation scenario with parameters"""
    name: str
    description: str
    stress_factors: Dict[str, float]  # Factor name -> intensity (0-1)
    duration: int  # Simulation steps
    critical_threshold: float = 0.6

class StressTestScenario(ABC):
    """Base class for stress test scenarios"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def apply_stress(self, model, data: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Apply stress to model and data"""
        pass
    
    @abstractmethod
    def measure_impact(self, original_results: Dict[str, Any], 
                      stressed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure impact of stress on trust metrics"""
        pass

class AdversarialAttackScenario(StressTestScenario):
    """Simulate adversarial attacks on the model"""
    
    def apply_stress(self, model, data: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Apply adversarial perturbations to data"""
        stressed_data = data.copy()
        
        # Add adversarial noise to prompts (simplified)
        if 'prompts' in stressed_data:
            prompts = stressed_data['prompts']
            stressed_prompts = []
            
            for prompt in prompts:
                if random.random() < intensity:  # Apply stress with probability
                    # Simple adversarial perturbation
                    perturbed_prompt = self._perturb_prompt(prompt, intensity)
                    stressed_prompts.append(perturbed_prompt)
                else:
                    stressed_prompts.append(prompt)
            
            stressed_data['prompts'] = stressed_prompts
        
        return stressed_data
    
    def _perturb_prompt(self, prompt: str, intensity: float) -> str:
        """Apply adversarial perturbation to prompt"""
        words = prompt.split()
        perturbation_count = max(1, int(len(words) * intensity * 0.3))
        
        for _ in range(perturbation_count):
            if words:
                # Randomly swap, delete, or add words
                action = random.choice(['swap', 'delete', 'add'])
                if action == 'swap' and len(words) > 1:
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
                elif action == 'delete' and len(words) > 1:
                    words.pop(random.randint(0, len(words) - 1))
                elif action == 'add':
                    words.insert(random.randint(0, len(words)), '[ATTACK]')
        
        return ' '.join(words)
    
    def measure_impact(self, original_results: Dict[str, Any], 
                      stressed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure impact of adversarial attack"""
        original_score = original_results.get('overall_trust_score', 0.5)
        stressed_score = stressed_results.get('overall_trust_score', 0.5)
        
        score_degradation = original_score - stressed_score
        
        # Analyze dimension-specific impacts
        original_dims = original_results.get('dimension_scores', {})
        stressed_dims = stressed_results.get('dimension_scores', {})
        
        dimension_impacts = {}
        for dim in set(original_dims.keys()) | set(stressed_dims.keys()):
            orig_val = original_dims.get(dim, 0.5)
            stress_val = stressed_dims.get(dim, 0.5)
            dimension_impacts[dim] = {
                'degradation': orig_val - stress_val,
                'percentage_drop': ((orig_val - stress_val) / orig_val * 100) if orig_val > 0 else 0
            }
        
        return {
            'score_degradation': score_degradation,
            'percentage_degradation': (score_degradation / original_score * 100) if original_score > 0 else 0,
            'dimension_impacts': dimension_impacts,
            'vulnerability_score': max(0, min(1, score_degradation * 2)),  # Scaled vulnerability
            'recommendation': self._generate_recommendation(score_degradation, dimension_impacts)
        }
    
    def _generate_recommendation(self, score_degradation: float, dimension_impacts: Dict) -> str:
        """Generate recommendation based on impact analysis"""
        if score_degradation > 0.3:
            return "High vulnerability to adversarial attacks. Implement robust adversarial training."
        elif score_degradation > 0.1:
            return "Moderate vulnerability detected. Consider adversarial defense mechanisms."
        else:
            return "Low vulnerability to tested adversarial scenarios."

class DataDriftScenario(StressTestScenario):
    """Simulate data drift scenarios"""
    
    def apply_stress(self, model, data: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Apply data drift simulation"""
        stressed_data = data.copy()
        
        # Simulate concept drift by modifying data characteristics
        if 'contexts' in stressed_data:
            contexts = stressed_data['contexts']
            drifted_contexts = []
            
            for context in contexts:
                if random.random() < intensity:
                    # Modify context to simulate drift
                    drifted_context = self._drift_context(context, intensity)
                    drifted_contexts.append(drifted_context)
                else:
                    drifted_contexts.append(context)
            
            stressed_data['contexts'] = drifted_contexts
        
        return stressed_data
    
    def _drift_context(self, context: str, intensity: float) -> str:
        """Apply context drift"""
        # Simplified context drift simulation
        drift_indicators = ['[FUTURE]', '[PAST]', '[DIFFERENT_DOMAIN]', '[EVOLVED]']
        drift_count = int(intensity * 3)
        
        for _ in range(drift_count):
            indicator = random.choice(drift_indicators)
            context = f"{indicator} {context}"
        
        return context
    
    def measure_impact(self, original_results: Dict[str, Any], 
                      stressed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure impact of data drift"""
        original_score = original_results.get('overall_trust_score', 0.5)
        stressed_score = stressed_results.get('overall_trust_score', 0.5)
        
        drift_impact = original_score - stressed_score
        
        return {
            'drift_impact': drift_impact,
            'adaptability_score': max(0, min(1, 1 - drift_impact)),  # Higher is better
            'recommendation': self._generate_recommendation(drift_impact)
        }
    
    def _generate_recommendation(self, drift_impact: float) -> str:
        """Generate recommendation based on drift impact"""
        if drift_impact > 0.2:
            return "Significant performance degradation under data drift. Implement continuous learning and monitoring."
        elif drift_impact > 0.1:
            return "Moderate drift sensitivity. Consider drift detection mechanisms."
        else:
            return "Good robustness to data drift scenarios."

class TrustSimulationEngine:
    """Main simulation engine for trust stress testing"""
    
    def __init__(self):
        self.scenarios = {
            'adversarial_attack': AdversarialAttackScenario(
                'adversarial_attack', 
                'Test model robustness against adversarial inputs'
            ),
            'data_drift': DataDriftScenario(
                'data_drift',
                'Test model performance under data distribution shifts'
            )
        }
        self.default_scenarios = [
            'adversarial_attack',
            'data_drift'
        ]
    
    def register_scenario(self, name: str, scenario: StressTestScenario):
        """Register a custom stress test scenario"""
        self.scenarios[name] = scenario
    
    def run_simulation(self, model, baseline_data: Dict[str, Any], 
                      scenario_name: str, intensity: float = 0.5,
                      evaluator = None) -> Dict[str, Any]:
        """Run a single simulation scenario"""
        if scenario_name not in self.scenarios:
            return {'error': f'Scenario {scenario_name} not found'}
        
        scenario = self.scenarios[scenario_name]
        
        # Get baseline evaluation
        if evaluator is None:
            from src.evaluators.composite_evaluator import CompositeTrustEvaluator
            evaluator = CompositeTrustEvaluator()
        
        baseline_results = evaluator.evaluate_comprehensive_trust(model, baseline_data)
        
        # Apply stress
        stressed_data = scenario.apply_stress(model, baseline_data, intensity)
        
        # Evaluate stressed performance
        stressed_results = evaluator.evaluate_comprehensive_trust(model, stressed_data)
        
        # Measure impact
        impact_analysis = scenario.measure_impact(baseline_results, stressed_results)
        
        return {
            'scenario': scenario_name,
            'intensity': intensity,
            'baseline_results': baseline_results,
            'stressed_results': stressed_results,
            'impact_analysis': impact_analysis,
            'stress_applied': stressed_data != baseline_data
        }
    
    def run_comprehensive_simulation(self, model, baseline_data: Dict[str, Any],
                                   scenarios: List[str] = None,
                                   intensities: List[float] = None,
                                   evaluator = None) -> Dict[str, Any]:
        """Run comprehensive simulation across multiple scenarios"""
        if scenarios is None:
            scenarios = self.default_scenarios
        
        if intensities is None:
            intensities = [0.3, 0.5, 0.7, 0.9]
        
        simulation_results = {}
        scenario_summaries = {}
        
        for scenario_name in scenarios:
            scenario_results = []
            for intensity in intensities:
                result = self.run_simulation(model, baseline_data, scenario_name, 
                                           intensity, evaluator)
                scenario_results.append(result)
            
            simulation_results[scenario_name] = scenario_results
            
            # Summarize scenario results
            scenario_summaries[scenario_name] = self._summarize_scenario_results(scenario_results)
        
        # Overall simulation summary
        overall_summary = self._generate_overall_summary(scenario_summaries)
        
        return {
            'detailed_results': simulation_results,
            'scenario_summaries': scenario_summaries,
            'overall_summary': overall_summary,
            'robustness_score': overall_summary.get('overall_robustness', 0.5),
            'recommendations': self._generate_simulation_recommendations(scenario_summaries)
        }
    
    def _summarize_scenario_results(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Summarize results for a single scenario across intensities"""
        if not scenario_results:
            return {}
        
        # Extract key metrics across intensities
        intensities = [r['intensity'] for r in scenario_results]
        impacts = [r['impact_analysis'].get('score_degradation', 0) for r in scenario_results]
        
        # Find maximum impact
        max_impact = max(impacts) if impacts else 0
        
        # Calculate robustness (inverse of impact)
        robustness_scores = [1 - impact for impact in impacts]
        avg_robustness = sum(robustness_scores) / len(robustness_scores) if robustness_scores else 0.5
        
        return {
            'max_impact': max_impact,
            'average_robustness': avg_robustness,
            'intensity_impact_curve': list(zip(intensities, impacts)),
            'worst_case_intensity': intensities[impacts.index(max_impact)] if impacts else 0.5
        }
    
    def _generate_overall_summary(self, scenario_summaries: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate overall simulation summary"""
        if not scenario_summaries:
            return {}
        
        avg_robustness = np.mean([summary.get('average_robustness', 0.5) 
                                 for summary in scenario_summaries.values()])
        
        max_impacts = [summary.get('max_impact', 0) 
                      for summary in scenario_summaries.values()]
        worst_case_impact = max(max_impacts) if max_impacts else 0
        
        return {
            'overall_robustness': float(avg_robustness),
            'worst_case_impact': worst_case_impact,
            'scenarios_tested': list(scenario_summaries.keys()),
            'deployment_readiness': 'HIGH' if avg_robustness > 0.8 else 
                                  'MEDIUM' if avg_robustness > 0.6 else 'LOW'
        }
    
    def _generate_simulation_recommendations(self, scenario_summaries: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        for scenario_name, summary in scenario_summaries.items():
            avg_robustness = summary.get('average_robustness', 0.5)
            max_impact = summary.get('max_impact', 0)
            
            if avg_robustness < 0.6:
                recommendations.append(f"Low robustness in {scenario_name} scenarios. Requires improvement.")
            elif max_impact > 0.3:
                recommendations.append(f"Significant vulnerability detected in {scenario_name}. Monitor closely.")
        
        return recommendations

# Integration with main evaluator
class SimulationEnhancedEvaluator:
    """Evaluator with simulation and stress testing capabilities"""
    
    def __init__(self):
        self.simulation_engine = TrustSimulationEngine()
        # ... other initialization
    
    def evaluate_with_simulation(self, model, data, 
                               simulation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute evaluation with comprehensive simulation testing"""
        
        # Standard evaluation
        standard_results = self.evaluate_comprehensive_trust(model, data)
        
        # Run simulations if configured
        if simulation_config is not None:
            simulation_results = self.simulation_engine.run_comprehensive_simulation(
                model, data, 
                scenarios=simulation_config.get('scenarios'),
                intensities=simulation_config.get('intensities'),
                evaluator=self  # Pass self as evaluator
            )
            
            # Combine results
            final_results = standard_results.copy()
            final_results['simulation_analysis'] = simulation_results
            
            # Add simulation-based trust score
            final_results['simulation_adjusted_trust'] = self._calculate_simulation_adjusted_trust(
                standard_results, simulation_results
            )
            
            return final_results
        else:
            return standard_results
    
    def _calculate_simulation_adjusted_trust(self, standard_results: Dict[str, Any], 
                                           simulation_results: Dict[str, Any]) -> float:
        """Calculate trust score adjusted for simulation results"""
        base_trust = standard_results.get('overall_trust_score', 0.5)
        simulation_robustness = simulation_results.get('robustness_score', 0.5)
        
        # Adjust trust score based on robustness
        adjusted_trust = base_trust * simulation_robustness
        
        return max(0, min(1, adjusted_trust))

# Usage example
def simulation_testing_example():
    """Example of trust simulation and stress testing"""
    
    # Create simulation-enhanced evaluator
    evaluator = SimulationEnhancedEvaluator()
    
    # Define simulation configuration
    simulation_config = {
        'scenarios': ['adversarial_attack', 'data_drift'],
        'intensities': [0.3, 0.5, 0.7, 0.9]
    }
    
    # Run evaluation with simulation
    results = evaluator.evaluate_with_simulation(
        model=my_llm_model,
        data=test_data,
        simulation_config=simulation_config
    )
    
    print("=== Trust Simulation Results ===")
    print(f"Base Trust Score: {results['overall_trust_score']:.3f}")
    print(f"Simulation-Adjusted Trust: {results['simulation_adjusted_trust']:.3f}")
    print(f"Overall Robustness: {results['simulation_analysis']['overall_summary']['overall_robustness']:.3f}")
    print(f"Deployment Readiness: {results['simulation_analysis']['overall_summary']['deployment_readiness']}")
    
    # Show scenario summaries
    print("\nScenario Summaries:")
    for scenario_name, summary in results['simulation_analysis']['scenario_summaries'].items():
        print(f"  {scenario_name}:")
        print(f"    Average Robustness: {summary['average_robustness']:.3f}")
        print(f"    Max Impact: {summary['max_impact']:.3f}")
    
    # Show recommendations
    if results['simulation_analysis']['recommendations']:
        print("\nRecommendations:")
        for rec in results['simulation_analysis']['recommendations']:
            print(f"  - {rec}")

# Advanced usage with custom scenarios
def custom_scenario_example():
    """Example with custom stress test scenarios"""
    
    class CustomStressScenario(StressTestScenario):
        def apply_stress(self, model, data: Dict[str, Any], intensity: float) -> Dict[str, Any]:
            # Custom stress logic
            stressed_data = data.copy()
            # ... implementation
            return stressed_data
        
        def measure_impact(self, original_results: Dict[str, Any], 
                          stressed_results: Dict[str, Any]) -> Dict[str, Any]:
            # Custom impact measurement
            return {'custom_impact': 0.5}
    
    # Register custom scenario
    simulation_engine = TrustSimulationEngine()
    simulation_engine.register_scenario('custom_stress', CustomStressScenario(
        'custom_stress', 'Custom stress test scenario'
    ))

  
# Use in evaluation
# ... implementation
