"""
Performance Monitoring and Analytics System
for Advanced Hallucination Detection

Features:
- Real-time performance metrics
- Predictive analytics
- Anomaly detection
- Performance optimization recommendations
- Historical trend analysis
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import redis
import logging
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: float
    metric_name: str
    value: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    timestamp: float
    metric_name: str
    severity: str
    description: str
    current_value: float
    expected_range: Tuple[float, float]
    recommendations: List[str]

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    timestamp: float
    summary: Dict[str, Any]
    metrics: Dict[str, List[float]]
    anomalies: List[AnomalyAlert]
    recommendations: List[str]
    trends: Dict[str, Any]

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_detectors = {}
        self.performance_thresholds = self._initialize_thresholds()
        
        # Real-time monitoring
        self.current_metrics = defaultdict(float)
        self.alert_history = deque(maxlen=100)
        
        # Performance optimization
        self.optimization_history = deque(maxlen=50)
        self.recommendation_engine = RecommendationEngine()
        
        # Analytics
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_analyzer = AnomalyAnalyzer()

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance thresholds"""
        return {
            'response_time': {
                'warning': 2.0,    # seconds
                'critical': 5.0,
                'optimal': 0.5
            },
            'throughput': {
                'warning': 50,     # requests per second
                'critical': 20,
                'optimal': 100
            },
            'error_rate': {
                'warning': 0.05,   # 5%
                'critical': 0.1,   # 10%
                'optimal': 0.01    # 1%
            },
            'cache_hit_rate': {
                'warning': 0.7,    # 70%
                'critical': 0.5,   # 50%
                'optimal': 0.9     # 90%
            },
            'memory_usage': {
                'warning': 0.8,    # 80%
                'critical': 0.95,  # 95%
                'optimal': 0.6     # 60%
            },
            'cpu_usage': {
                'warning': 0.7,    # 70%
                'critical': 0.9,   # 90%
                'optimal': 0.5     # 50%
            }
        }

    def record_metric(self, metric_name: str, value: float, source: str = "system", metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        timestamp = time.time()
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            source=source,
            metadata=metadata or {}
        )
        
        # Store in buffer
        self.metrics_buffer[metric_name].append(metric)
        
        # Update current metrics
        self.current_metrics[metric_name] = value
        
        # Check for anomalies
        self._check_anomalies(metric)
        
        # Store in Redis for persistence
        self._store_metric(metric)

    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in Redis"""
        try:
            key = f"metric:{metric.metric_name}:{int(metric.timestamp)}"
            data = {
                'timestamp': metric.timestamp,
                'value': metric.value,
                'source': metric.source,
                'metadata': metric.metadata
            }
            self.redis.setex(key, 86400, json.dumps(data))  # 24 hour TTL
        except Exception as e:
            logger.warning(f"Failed to store metric: {e}")

    def _check_anomalies(self, metric: PerformanceMetric):
        """Check for anomalies in the metric"""
        if metric.metric_name not in self.anomaly_detectors:
            self.anomaly_detectors[metric.metric_name] = AnomalyDetector()
        
        detector = self.anomaly_detectors[metric.metric_name]
        is_anomaly = detector.detect_anomaly(metric.value)
        
        if is_anomaly:
            alert = self._create_anomaly_alert(metric)
            self.alert_history.append(alert)
            logger.warning(f"Anomaly detected: {alert.description}")

    def _create_anomaly_alert(self, metric: PerformanceMetric) -> AnomalyAlert:
        """Create anomaly alert"""
        thresholds = self.performance_thresholds.get(metric.metric_name, {})
        
        if metric.value > thresholds.get('critical', float('inf')):
            severity = 'critical'
        elif metric.value > thresholds.get('warning', float('inf')):
            severity = 'warning'
        else:
            severity = 'info'
        
        expected_range = (
            thresholds.get('optimal', 0),
            thresholds.get('warning', float('inf'))
        )
        
        recommendations = self.recommendation_engine.get_recommendations(
            metric.metric_name, metric.value, severity
        )
        
        return AnomalyAlert(
            timestamp=metric.timestamp,
            metric_name=metric.metric_name,
            severity=severity,
            description=f"{metric.metric_name} value {metric.value} exceeds {severity} threshold",
            current_value=metric.value,
            expected_range=expected_range,
            recommendations=recommendations
        )

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return dict(self.current_metrics)

    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metric history for the specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        # Get from buffer
        metrics = [
            metric for metric in self.metrics_buffer[metric_name]
            if metric.timestamp >= cutoff_time
        ]
        
        # Get from Redis if needed
        if len(metrics) < 100:  # If buffer doesn't have enough data
            redis_metrics = self._get_metrics_from_redis(metric_name, cutoff_time)
            metrics.extend(redis_metrics)
        
        return sorted(metrics, key=lambda x: x.timestamp)

    def _get_metrics_from_redis(self, metric_name: str, cutoff_time: float) -> List[PerformanceMetric]:
        """Get metrics from Redis"""
        metrics = []
        try:
            pattern = f"metric:{metric_name}:*"
            keys = self.redis.keys(pattern)
            
            for key in keys:
                data = self.redis.get(key)
                if data:
                    metric_data = json.loads(data)
                    if metric_data['timestamp'] >= cutoff_time:
                        metric = PerformanceMetric(
                            timestamp=metric_data['timestamp'],
                            metric_name=metric_name,
                            value=metric_data['value'],
                            source=metric_data['source'],
                            metadata=metric_data['metadata']
                        )
                        metrics.append(metric)
        except Exception as e:
            logger.warning(f"Failed to get metrics from Redis: {e}")
        
        return metrics

    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        timestamp = time.time()
        
        # Collect metrics
        all_metrics = {}
        for metric_name in self.metrics_buffer.keys():
            history = self.get_metric_history(metric_name, hours)
            all_metrics[metric_name] = [m.value for m in history]
        
        # Analyze trends
        trends = self.trend_analyzer.analyze_trends(all_metrics)
        
        # Detect anomalies
        anomalies = list(self.alert_history)[-10:]  # Last 10 alerts
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            all_metrics, trends, anomalies
        )
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(all_metrics)
        
        return PerformanceReport(
            timestamp=timestamp,
            summary=summary,
            metrics=all_metrics,
            anomalies=anomalies,
            recommendations=recommendations,
            trends=trends
        )

    def _calculate_summary_statistics(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics"""
        summary = {}
        
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        current_metrics = self.get_current_metrics()
        
        # Calculate health scores
        health_scores = {}
        for metric_name, value in current_metrics.items():
            thresholds = self.performance_thresholds.get(metric_name, {})
            optimal = thresholds.get('optimal', 0)
            warning = thresholds.get('warning', float('inf'))
            
            if value <= optimal:
                health_scores[metric_name] = 1.0
            elif value <= warning:
                health_scores[metric_name] = 0.5
            else:
                health_scores[metric_name] = 0.0
        
        # Get recent alerts
        recent_alerts = list(self.alert_history)[-5:]
        
        # Get optimization history
        recent_optimizations = list(self.optimization_history)[-5:]
        
        return {
            'current_metrics': current_metrics,
            'health_scores': health_scores,
            'recent_alerts': recent_alerts,
            'recent_optimizations': recent_optimizations,
            'overall_health': np.mean(list(health_scores.values())) if health_scores else 0.0
        }

    def record_optimization(self, optimization_type: str, description: str, impact: Dict[str, float]):
        """Record performance optimization"""
        optimization = {
            'timestamp': time.time(),
            'type': optimization_type,
            'description': description,
            'impact': impact
        }
        
        self.optimization_history.append(optimization)
        logger.info(f"Performance optimization recorded: {description}")

class AnomalyDetector:
    """Anomaly detection using isolation forest"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.values_buffer = deque(maxlen=100)

    def detect_anomaly(self, value: float) -> bool:
        """Detect if a value is anomalous"""
        self.values_buffer.append(value)
        
        if len(self.values_buffer) < 10:
            return False  # Need more data
        
        # Convert to numpy array
        values = np.array(list(self.values_buffer)).reshape(-1, 1)
        
        if not self.is_fitted:
            # Fit the model
            self.scaler.fit(values)
            scaled_values = self.scaler.transform(values)
            self.model.fit(scaled_values)
            self.is_fitted = True
            return False
        
        # Predict anomaly
        scaled_value = self.scaler.transform([[value]])
        prediction = self.model.predict(scaled_value)
        
        return prediction[0] == -1  # -1 indicates anomaly

class TrendAnalyzer:
    """Analyze trends in performance metrics"""
    
    def analyze_trends(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze trends in metrics"""
        trends = {}
        
        for metric_name, values in metrics.items():
            if len(values) < 2:
                continue
            
            # Calculate trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Determine trend direction
            if slope > 0.01:
                direction = 'increasing'
            elif slope < -0.01:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Calculate volatility
            volatility = np.std(values)
            
            # Detect seasonality (simple approach)
            seasonality = self._detect_seasonality(values)
            
            trends[metric_name] = {
                'direction': direction,
                'slope': slope,
                'volatility': volatility,
                'seasonality': seasonality,
                'trend_strength': abs(slope) / (volatility + 1e-6)
            }
        
        return trends

    def _detect_seasonality(self, values: List[float]) -> str:
        """Detect seasonality in time series"""
        if len(values) < 20:
            return 'insufficient_data'
        
        # Simple seasonality detection using autocorrelation
        values_array = np.array(values)
        autocorr = np.correlate(values_array, values_array, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        if len(peaks) > 0:
            avg_period = np.mean(peaks)
            if avg_period < 5:
                return 'high_frequency'
            elif avg_period < 20:
                return 'medium_frequency'
            else:
                return 'low_frequency'
        
        return 'no_seasonality'

class AnomalyAnalyzer:
    """Analyze patterns in anomalies"""
    
    def analyze_anomaly_patterns(self, alerts: List[AnomalyAlert]) -> Dict[str, Any]:
        """Analyze patterns in anomaly alerts"""
        if not alerts:
            return {}
        
        # Group by metric
        metric_groups = defaultdict(list)
        for alert in alerts:
            metric_groups[alert.metric_name].append(alert)
        
        patterns = {}
        for metric_name, metric_alerts in metric_groups.items():
            # Analyze frequency
            time_range = max(alert.timestamp for alert in metric_alerts) - min(alert.timestamp for alert in metric_alerts)
            frequency = len(metric_alerts) / (time_range / 3600) if time_range > 0 else 0  # alerts per hour
            
            # Analyze severity distribution
            severity_counts = defaultdict(int)
            for alert in metric_alerts:
                severity_counts[alert.severity] += 1
            
            # Analyze time patterns
            hour_distribution = defaultdict(int)
            for alert in metric_alerts:
                hour = datetime.fromtimestamp(alert.timestamp).hour
                hour_distribution[hour] += 1
            
            patterns[metric_name] = {
                'frequency': frequency,
                'severity_distribution': dict(severity_counts),
                'hour_distribution': dict(hour_distribution),
                'total_alerts': len(metric_alerts)
            }
        
        return patterns

class RecommendationEngine:
    """Generate performance optimization recommendations"""
    
    def get_recommendations(self, metric_name: str, value: float, severity: str) -> List[str]:
        """Get recommendations for a specific metric"""
        recommendations = []
        
        if metric_name == 'response_time':
            if severity == 'critical':
                recommendations.extend([
                    "Consider horizontal scaling of verification services",
                    "Implement connection pooling for external APIs",
                    "Optimize database queries and add indexes",
                    "Review and optimize verification algorithms"
                ])
            elif severity == 'warning':
                recommendations.extend([
                    "Monitor external API response times",
                    "Consider implementing caching for frequently requested data",
                    "Review verification source priorities"
                ])
        
        elif metric_name == 'error_rate':
            if severity == 'critical':
                recommendations.extend([
                    "Implement circuit breakers for failing services",
                    "Add retry mechanisms with exponential backoff",
                    "Review error handling and logging",
                    "Consider fallback verification sources"
                ])
            elif severity == 'warning':
                recommendations.extend([
                    "Monitor error patterns and implement fixes",
                    "Add health checks for verification services",
                    "Review API rate limits and quotas"
                ])
        
        elif metric_name == 'cache_hit_rate':
            if severity == 'critical':
                recommendations.extend([
                    "Increase cache TTL for frequently accessed data",
                    "Implement cache warming strategies",
                    "Review cache eviction policies",
                    "Consider distributed caching"
                ])
            elif severity == 'warning':
                recommendations.extend([
                    "Optimize cache key generation",
                    "Review cache size and memory allocation",
                    "Implement cache analytics"
                ])
        
        elif metric_name == 'memory_usage':
            if severity == 'critical':
                recommendations.extend([
                    "Implement memory leak detection",
                    "Optimize data structures and algorithms",
                    "Consider garbage collection tuning",
                    "Review memory allocation patterns"
                ])
            elif severity == 'warning':
                recommendations.extend([
                    "Monitor memory usage patterns",
                    "Implement memory usage alerts",
                    "Review object lifecycle management"
                ])
        
        elif metric_name == 'cpu_usage':
            if severity == 'critical':
                recommendations.extend([
                    "Implement CPU profiling and optimization",
                    "Consider parallel processing for verification tasks",
                    "Review algorithm complexity",
                    "Implement request throttling"
                ])
            elif severity == 'warning':
                recommendations.extend([
                    "Monitor CPU usage patterns",
                    "Implement CPU usage alerts",
                    "Review processing efficiency"
                ])
        
        return recommendations

    def generate_recommendations(self, metrics: Dict[str, List[float]], trends: Dict[str, Any], anomalies: List[AnomalyAlert]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Analyze trends and generate recommendations
        for metric_name, trend_data in trends.items():
            if trend_data['direction'] == 'increasing' and trend_data['trend_strength'] > 0.5:
                if 'response_time' in metric_name:
                    recommendations.append(f"Response time for {metric_name} is trending upward. Consider optimization.")
                elif 'error_rate' in metric_name:
                    recommendations.append(f"Error rate for {metric_name} is increasing. Investigate root cause.")
                elif 'memory_usage' in metric_name:
                    recommendations.append(f"Memory usage for {metric_name} is growing. Check for memory leaks.")
        
        # Analyze anomalies
        if len(anomalies) > 5:
            recommendations.append("High number of anomalies detected. Review system stability.")
        
        # Check for performance bottlenecks
        for metric_name, values in metrics.items():
            if values and np.mean(values) > 0.8:  # High utilization
                recommendations.append(f"High utilization detected for {metric_name}. Consider scaling.")
        
        return list(set(recommendations))  # Remove duplicates

class PerformanceVisualizer:
    """Visualize performance metrics"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use(self.style)
    
    def plot_metric_trends(self, metrics: Dict[str, List[float]], title: str = "Performance Metrics"):
        """Plot metric trends"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        metric_names = list(metrics.keys())[:4]  # Plot first 4 metrics
        
        for i, metric_name in enumerate(metric_names):
            row = i // 2
            col = i % 2
            
            values = metrics[metric_name]
            if values:
                axes[row, col].plot(values, marker='o', alpha=0.7)
                axes[row, col].set_title(metric_name)
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel('Value')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_health_dashboard(self, health_scores: Dict[str, float]):
        """Plot health dashboard"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Health scores bar chart
        metrics = list(health_scores.keys())
        scores = list(health_scores.values())
        
        colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in scores]
        
        ax1.bar(metrics, scores, color=colors, alpha=0.7)
        ax1.set_title('System Health Scores')
        ax1.set_ylabel('Health Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Overall health gauge
        overall_health = np.mean(scores) if scores else 0
        ax2.pie([overall_health, 1-overall_health], 
                labels=['Healthy', 'Unhealthy'],
                colors=['green', 'red'],
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title('Overall System Health')
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_timeline(self, alerts: List[AnomalyAlert]):
        """Plot anomaly timeline"""
        if not alerts:
            return None
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        timestamps = [alert.timestamp for alert in alerts]
        severities = [alert.severity for alert in alerts]
        
        # Color coding for severity
        colors = {'critical': 'red', 'warning': 'orange', 'info': 'blue'}
        color_values = [colors.get(severity, 'gray') for severity in severities]
        
        ax.scatter(timestamps, range(len(timestamps)), c=color_values, s=100, alpha=0.7)
        ax.set_title('Anomaly Timeline')
        ax.set_xlabel('Time')
        ax.set_ylabel('Alert Index')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, label=severity, markersize=10)
                          for severity, color in colors.items()]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig 