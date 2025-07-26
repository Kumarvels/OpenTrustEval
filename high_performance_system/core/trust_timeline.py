# src/evolution/trust_timeline.py
"""
Trust Evolution System - Track how trust changes over time, contexts, and model updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px

class TrustEvolutionTracker:
    """Tracks trust evolution across multiple dimensions and time periods"""
    
    def __init__(self):
        self.trust_history = pd.DataFrame()
        self.evolution_patterns = {}
        self.anomaly_detectors = {}
    
    def track_evaluation(self, model_id: str, evaluation_results: Dict[str, Any], 
                        context: str = "general", timestamp: datetime = None):
        """Track a trust evaluation in the timeline"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract trust metrics
        record = {
            'model_id': model_id,
            'timestamp': timestamp,
            'context': context,
            'overall_trust': evaluation_results.get('overall_trust_score', 0.0),
            'dimensions': evaluation_results.get('dimension_scores', {}),
            'categories': evaluation_results.get('category_scores', {}),
            'metadata': evaluation_results.get('metadata', {})
        }
        
        # Add to history
        self.trust_history = pd.concat([self.trust_history, pd.DataFrame([record])], 
                                     ignore_index=True)
        
        # Update evolution patterns
        self._update_evolution_patterns(model_id, record)
    
    def detect_trust_anomalies(self, model_id: str) -> List[Dict[str, Any]]:
        """Detect anomalies in trust evolution"""
        model_history = self.trust_history[self.trust_history['model_id'] == model_id]
        
        anomalies = []
        for dimension in ['overall_trust'] + list(model_history['dimensions'].iloc[0].keys()):
            scores = model_history[dimension].values if dimension == 'overall_trust' else \
                    [d.get(dimension, 0) for d in model_history['dimensions']]
            
            # Statistical anomaly detection
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Check for recent significant drops
            if len(scores) > 3:
                recent_drop = scores[-1] < (mean_score - 2 * std_score)
                if recent_drop:
                    anomalies.append({
                        'type': 'significant_drop',
                        'dimension': dimension,
                        'current_score': scores[-1],
                        'historical_mean': mean_score,
                        'severity': 'high' if scores[-1] < (mean_score - 3 * std_score) else 'medium',
                        'timestamp': model_history.iloc[-1]['timestamp']
                    })
        
        return anomalies
    
    def predict_trust_trajectory(self, model_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future trust trajectory using time series analysis"""
        model_history = self.trust_history[self.trust_history['model_id'] == model_id]
        
        if len(model_history) < 5:
            return {'status': 'insufficient_data', 'prediction': None}
        
        # Simple linear regression for prediction
        timestamps = [(t - model_history.iloc[0]['timestamp']).days 
                     for t in model_history['timestamp']]
        overall_scores = model_history['overall_trust'].values
        
        # Linear regression
        if len(set(timestamps)) > 1:  # Avoid division by zero
            slope = np.polyfit(timestamps, overall_scores, 1)[0]
            predicted_score = overall_scores[-1] + (slope * days_ahead)
            
            return {
                'status': 'success',
                'current_score': overall_scores[-1],
                'predicted_score': max(0, min(1, predicted_score)),  # Clamp between 0-1
                'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                'confidence': min(1.0, len(timestamps) / 20.0)  # Confidence increases with more data
            }
        
        return {'status': 'insufficient_variance', 'prediction': None}
    
    def generate_evolution_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive trust evolution report"""
        model_history = self.trust_history[self.trust_history['model_id'] == model_id]
        
        if len(model_history) == 0:
            return {'status': 'no_data', 'report': None}
        
        # Calculate evolution metrics
        first_score = model_history.iloc[0]['overall_trust']
        last_score = model_history.iloc[-1]['overall_trust']
        score_change = last_score - first_score
        
        # Dimension evolution
        dimension_evolution = {}
        if len(model_history) > 1:
            first_dims = model_history.iloc[0]['dimensions']
            last_dims = model_history.iloc[-1]['dimensions']
            
            for dim in set(first_dims.keys()) | set(last_dims.keys()):
                first_val = first_dims.get(dim, 0)
                last_val = last_dims.get(dim, 0)
                dimension_evolution[dim] = {
                    'change': last_val - first_val,
                    'percentage_change': ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                }
        
        return {
            'status': 'success',
            'model_id': model_id,
            'evaluation_count': len(model_history),
            'time_span': (model_history.iloc[-1]['timestamp'] - model_history.iloc[0]['timestamp']).days,
            'overall_evolution': {
                'initial_score': first_score,
                'current_score': last_score,
                'absolute_change': score_change,
                'percentage_change': (score_change / first_score * 100) if first_score > 0 else 0
            },
            'dimension_evolution': dimension_evolution,
            'anomalies_detected': self.detect_trust_anomalies(model_id),
            'trajectory_prediction': self.predict_trust_trajectory(model_id)
        }

# Integration with main evaluator
class EvolutionAwareEvaluator:
    """Evaluator that tracks trust evolution"""
    
    def __init__(self):
        self.trust_tracker = TrustEvolutionTracker()
        # ... other initialization
    
    def evaluate_with_evolution_tracking(self, model, data, model_id: str, 
                                       context: str = "general") -> Dict[str, Any]:
        """Execute evaluation with evolution tracking"""
        results = self.evaluate_comprehensive_trust(model, data)
        
        # Track in evolution system
        self.trust_tracker.track_evaluation(model_id, results, context)
        
        # Add evolution insights
        results['evolution_insights'] = {
            'anomalies': self.trust_tracker.detect_trust_anomalies(model_id),
            'trajectory': self.trust_tracker.predict_trust_trajectory(model_id),
            'report': self.trust_tracker.generate_evolution_report(model_id)
        }
        
        return results
