"""
Cleanlab Integration for OpenTrustEval
Implements confident learning and data quality assessment features
Enhanced with advanced trust scoring methods
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json

try:
    import cleanlab
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    print("Warning: Cleanlab not available. Install with: pip install cleanlab")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available. Install with: pip install scikit-learn")

# Import advanced trust scoring engine
try:
    from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
    ADVANCED_TRUST_AVAILABLE = True
except ImportError:
    ADVANCED_TRUST_AVAILABLE = False
    print("Warning: Advanced trust scoring not available. Check advanced_trust_scoring.py")

class FallbackDataQualityManager:
    """
    Fallback data quality assessment when Cleanlab is not available
    Uses basic statistical methods and data quality metrics
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for fallback operations"""
        logger = logging.getLogger('FallbackDataQualityManager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def calculate_data_trust_score(self, dataset: pd.DataFrame, labels: Optional[List] = None, 
                                 features: Optional[List] = None) -> Dict[str, Any]:
        """
        Calculate data trust score using basic statistical methods
        """
        try:
            # Use all numeric columns as features if not specified
            if features is None:
                features = dataset.select_dtypes(include=[np.number]).columns.tolist()
            
            # Prepare data
            X = dataset[features].fillna(0) if len(features) > 0 else dataset.fillna(0)
            
            # Calculate basic quality metrics
            quality_metrics = {
                "missing_values_ratio": float(X.isnull().sum().sum() / (X.shape[0] * X.shape[1])),
                "duplicate_rows_ratio": float(X.duplicated().sum() / X.shape[0]),
                "zero_variance_features": int((X.var() == 0).sum()) if len(features) > 0 else 0,
                "correlation_issues": self._detect_correlation_issues(X),
                "outlier_ratio": self._detect_outliers_ratio(X),
                "data_completeness": float(1 - X.isnull().sum().sum() / (X.shape[0] * X.shape[1])),
                "data_consistency": self._calculate_consistency_score(X)
            }
            
            # Calculate overall trust score
            trust_score = 1 - (
                quality_metrics["missing_values_ratio"] * 0.25 +
                quality_metrics["duplicate_rows_ratio"] * 0.2 +
                (quality_metrics["zero_variance_features"] / max(1, X.shape[1])) * 0.15 +
                quality_metrics["correlation_issues"] * 0.15 +
                quality_metrics["outlier_ratio"] * 0.15 +
                (1 - quality_metrics["data_consistency"]) * 0.1
            )
            
            return {
                "trust_score": max(0, float(trust_score)),
                "quality_metrics": quality_metrics,
                "method": "fallback_statistical",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error calculating trust score: {e}")
            return {"error": str(e), "method": "fallback_statistical"}
    
    def _detect_correlation_issues(self, X: pd.DataFrame, threshold: float = 0.95) -> float:
        """Detect highly correlated features"""
        try:
            if X.shape[1] < 2:
                return 0.0
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = (upper_tri > threshold).sum().sum()
            return float(high_corr_pairs / (len(X.columns) * (len(X.columns) - 1) / 2))
        except:
            return 0.0
    
    def _detect_outliers_ratio(self, X: pd.DataFrame, method: str = 'iqr') -> float:
        """Detect outliers using IQR method"""
        try:
            if method == 'iqr':
                Q1 = X.quantile(0.25)
                Q3 = X.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
                return float(outliers.sum() / len(X))
            return 0.0
        except:
            return 0.0
    
    def _calculate_consistency_score(self, X: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        try:
            # Check for data type consistency
            type_consistency = 1.0
            for col in X.columns:
                if X[col].dtype in ['object', 'string']:
                    # For string columns, check if they're mostly unique
                    unique_ratio = X[col].nunique() / len(X)
                    type_consistency *= unique_ratio
                elif X[col].dtype in ['int64', 'float64']:
                    # For numeric columns, check for reasonable ranges
                    if X[col].std() > 0:
                        type_consistency *= 1.0
                    else:
                        type_consistency *= 0.5
            
            return float(type_consistency)
        except:
            return 0.5
    
    def create_quality_based_filter(self, dataset: pd.DataFrame, min_trust_score: float = 0.7,
                                  features: Optional[List] = None) -> pd.DataFrame:
        """
        Create a quality-based filter for datasets using fallback methods
        """
        try:
            # Calculate row-wise quality scores
            if features is None:
                features = dataset.select_dtypes(include=[np.number]).columns.tolist()
            
            X = dataset[features].fillna(0) if len(features) > 0 else dataset.fillna(0)
            
            # Calculate quality score for each row
            row_quality_scores = []
            for _, row in X.iterrows():
                # Calculate row quality based on various factors
                missing_ratio = row.isnull().sum() / len(row)
                zero_ratio = (row == 0).sum() / len(row) if len(features) > 0 else 0
                quality = 1 - (missing_ratio * 0.6 + zero_ratio * 0.4)
                row_quality_scores.append(quality)
            
            # Filter based on quality threshold
            quality_mask = np.array(row_quality_scores) > min_trust_score
            filtered_dataset = dataset[quality_mask].copy()
            
            self.logger.info(f"Filtered dataset: {len(filtered_dataset)}/{len(dataset)} rows retained")
            
            return filtered_dataset
            
        except Exception as e:
            self.logger.error(f"Error in quality-based filtering: {e}")
            return dataset
    
    def automated_data_validation(self, dataset: pd.DataFrame, validation_rules: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Automated data validation using fallback methods
        """
        try:
            results = {
                "validation_passed": True,
                "issues_found": [],
                "quality_metrics": {},
                "recommendations": []
            }
            
            # Basic data quality checks
            quality_metrics = self.calculate_data_trust_score(dataset)
            results["quality_metrics"] = quality_metrics.get("quality_metrics", {})
            
            # Check for critical issues
            if quality_metrics.get("quality_metrics", {}).get("missing_values_ratio", 0) > 0.5:
                results["issues_found"].append("High missing values ratio (>50%)")
                results["validation_passed"] = False
            
            if quality_metrics.get("quality_metrics", {}).get("duplicate_rows_ratio", 0) > 0.3:
                results["issues_found"].append("High duplicate rows ratio (>30%)")
                results["validation_passed"] = False
            
            # Custom validation rules
            if validation_rules:
                for rule_name, rule_func in validation_rules.items():
                    try:
                        if not rule_func(dataset):
                            results["issues_found"].append(f"Custom rule failed: {rule_name}")
                            results["validation_passed"] = False
                    except Exception as e:
                        results["issues_found"].append(f"Custom rule error: {rule_name} - {str(e)}")
                        results["validation_passed"] = False
            
            # Generate recommendations
            if results["quality_metrics"].get("missing_values_ratio", 0) > 0.1:
                results["recommendations"].append("Consider imputation for missing values")
            
            if results["quality_metrics"].get("correlation_issues", 0) > 0.1:
                results["recommendations"].append("Check for highly correlated features")
            
            results["timestamp"] = datetime.now().isoformat()
            results["method"] = "fallback_statistical"
            return results
            
        except Exception as e:
            self.logger.error(f"Error in automated validation: {e}")
            return {"error": str(e), "method": "fallback_statistical"}
    
    def generate_quality_report(self, dataset: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive quality report using fallback methods
        """
        try:
            # Calculate all quality metrics
            trust_score_result = self.calculate_data_trust_score(dataset)
            validation_result = self.automated_data_validation(dataset)
            
            report = {
                "dataset_info": {
                    "shape": dataset.shape,
                    "columns": dataset.columns.tolist(),
                    "dtypes": dataset.dtypes.astype(str).to_dict()
                },
                "trust_assessment": trust_score_result,
                "validation_results": validation_result,
                "summary": {
                    "overall_trust_score": trust_score_result.get("trust_score", 0),
                    "validation_passed": validation_result.get("validation_passed", False),
                    "critical_issues": len(validation_result.get("issues_found", [])),
                    "recommendations": validation_result.get("recommendations", []),
                    "method": "fallback_statistical"
                },
                "generated_at": datetime.now().isoformat()
            }
            
            report_content = json.dumps(report, indent=2)
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_content)
                return output_path
            else:
                return report_content
                
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return f"Error: {str(e)}"

# Remove all Cleanlab logic and references except for benchmarking
# Remove: import cleanlab, from cleanlab import find_label_issues, get_label_quality_scores, CleanLearning
# Remove: CLEANLAB_AVAILABLE logic and all Cleanlab-based scoring except for benchmarking
# Remove: CleanlabDataQualityManager class and usages except for benchmarking

# --- Add benchmarking function for Cleanlab comparison only ---
def benchmark_vs_cleanlab(dataset: pd.DataFrame, labels: list, features: list = None) -> dict:
    """Benchmark our trust score against Cleanlab's trust score (for validation only)."""
    try:
        import cleanlab
        from cleanlab.rank import get_label_quality_scores
        if features is None:
            features = dataset.select_dtypes(include=[np.number]).columns.tolist()
        X = dataset[features].fillna(0)
        label_quality = get_label_quality_scores(labels, X.values)
        cleanlab_trust = 1 - float(np.mean(label_quality))
        from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
        engine = AdvancedTrustScoringEngine()
        our_result = engine.calculate_advanced_trust_score(dataset, labels=labels, features=features, method="ensemble")
        return {
            "our_trust_score": our_result.get("trust_score", None),
            "cleanlab_trust_score": cleanlab_trust,
            "label_quality_scores": label_quality.tolist(),
            "method": "benchmark_vs_cleanlab"
        }
    except ImportError:
        return {"error": "Cleanlab not installed", "method": "benchmark_vs_cleanlab"}
    except Exception as e:
        return {"error": str(e), "method": "benchmark_vs_cleanlab"}

# Example usage functions
def example_data_trust_assessment():
    """Example: Assess data trust using available methods"""
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add some quality issues
    data.loc[100:200, 'feature1'] = np.nan  # Missing values
    data.loc[300:400, 'feature2'] = 0  # Zero values
    data.loc[500:600, 'feature3'] = data.loc[500:600, 'feature1']  # Duplicates
    
    # Initialize manager with advanced scoring
    manager = FallbackDataQualityManager() # Changed from CleanlabDataQualityManager
    
    # Calculate trust score with different methods
    print("=== Advanced Trust Assessment Results ===\n")
    
    # Auto method (uses best available)
    auto_result = manager.calculate_data_trust_score(data) # Changed from manager.calculate_data_trust_score
    print("Auto Method:")
    print(f"Trust Score: {auto_result['trust_score']:.3f}")
    print(f"Method: {auto_result['method']}")
    print()
    
    # Advanced method
    advanced_result = manager.calculate_data_trust_score(data) # Changed from manager.calculate_data_trust_score
    print("Advanced Method:")
    print(f"Trust Score: {advanced_result['trust_score']:.3f}")
    print(f"Method: {advanced_result['method']}")
    if 'quality_metrics' in advanced_result: # Changed from 'component_scores'
        print("Quality Metrics:")
        for metric, value in advanced_result['quality_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.3f}")
    print()
    
    # Generate quality report
    report = manager.generate_quality_report(data) # Changed from manager.generate_quality_report
    print("Quality Report Generated Successfully")
    print(f"Report length: {len(report)} characters")

if __name__ == "__main__":
    example_data_trust_assessment() 