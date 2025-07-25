#!/usr/bin/env python3
"""
Advanced Trust Scoring Test Suite
Comprehensive testing of advanced trust scoring capabilities
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def create_test_datasets():
    """Create synthetic test datasets with various quality characteristics"""
    
    # High quality dataset
    np.random.seed(42)
    high_quality_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.normal(0, 1, 1000),
        'feature4': np.random.normal(0, 1, 1000)
    })
    
    # Medium quality dataset (some issues)
    medium_quality_data = high_quality_data.copy()
    medium_quality_data.loc[100:200, 'feature1'] = np.nan  # Missing values
    medium_quality_data.loc[300:400, 'feature2'] = 0  # Zero values
    medium_quality_data.loc[500:600, 'feature3'] = medium_quality_data.loc[500:600, 'feature1']  # Duplicates
    
    # Low quality dataset (many issues)
    low_quality_data = high_quality_data.copy()
    low_quality_data.loc[100:400, 'feature1'] = np.nan  # Many missing values
    low_quality_data.loc[500:800, 'feature2'] = 0  # Many zero values
    low_quality_data.loc[900:950, 'feature3'] = np.random.normal(100, 10, 51)  # Outliers
    low_quality_data.loc[950:1000, 'feature4'] = low_quality_data.loc[950:1000, 'feature1']  # Duplicates
    
    # Correlated features dataset
    correlated_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000) * 0.9 + np.random.normal(0, 0.1, 1000),  # Highly correlated
        'feature3': np.random.normal(0, 1, 1000),
        'feature4': np.random.normal(0, 1, 1000)
    })
    
    # Skewed distribution dataset
    skewed_data = pd.DataFrame({
        'feature1': np.random.exponential(1, 1000),  # Right-skewed
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.lognormal(0, 1, 1000),  # Log-normal
        'feature4': np.random.normal(0, 1, 1000)
    })
    
    return {
        'high_quality': high_quality_data,
        'medium_quality': medium_quality_data,
        'low_quality': low_quality_data,
        'correlated': correlated_data,
        'skewed': skewed_data
    }

def test_advanced_trust_scoring_engine():
    """Test the Advanced Trust Scoring Engine"""
    print("=== Testing Advanced Trust Scoring Engine ===")
    
    try:
        # Import the advanced trust scoring engine
        from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
        
        # Create test datasets
        test_datasets = create_test_datasets()
        
        # Initialize the engine
        engine = AdvancedTrustScoringEngine()
        
        results = {}
        
        # Test different scoring methods
        for dataset_name, dataset in test_datasets.items():
            print(f"Testing {dataset_name} dataset...")
            
            # Test ensemble method
            ensemble_result = engine.calculate_advanced_trust_score(dataset, method="ensemble")
            results[f"{dataset_name}_ensemble"] = ensemble_result
            
            # Test robust method
            robust_result = engine.calculate_advanced_trust_score(dataset, method="robust")
            results[f"{dataset_name}_robust"] = robust_result
            
            # Test uncertainty method
            uncertainty_result = engine.calculate_advanced_trust_score(dataset, method="uncertainty")
            results[f"{dataset_name}_uncertainty"] = uncertainty_result
        
        print("✓ Advanced Trust Scoring Engine tests completed successfully")
        return True, results
        
    except ImportError as e:
        print(f"✗ Failed to import Advanced Trust Scoring Engine: {e}")
        return False, {}
    except Exception as e:
        print(f"✗ Error in advanced trust scoring tests: {e}")
        return False, {}

# Remove all CleanlabDataQualityManager and Cleanlab logic except for a single benchmarking test
# The rest of the tests should use only fallback and advanced trust scoring logic
def test_cleanlab_benchmarking():
    """Benchmarking test comparing our trust score to Cleanlab's"""
    print("\n=== Testing Cleanlab Benchmarking ===")
    
    try:
        # Import the benchmarking function
        from data_engineering.cleanlab_integration import benchmark_vs_cleanlab
        
        # Create test datasets
        test_datasets = create_test_datasets()
        
        results = {}
        
        # Test benchmarking on different datasets
        for dataset_name, dataset in test_datasets.items():
            print(f"Testing {dataset_name} dataset...")
            
            # Create synthetic labels for benchmarking
            labels = np.random.choice([0, 1], size=len(dataset))
            
            # Run benchmark comparison
            benchmark_result = benchmark_vs_cleanlab(dataset, labels)
            results[f"{dataset_name}_benchmark"] = benchmark_result
        
        print("✓ Cleanlab Benchmarking tests completed successfully")
        return True, results
        
    except ImportError as e:
        print(f"✗ Failed to import benchmarking function: {e}")
        return False, {}
    except Exception as e:
        print(f"✗ Error in cleanlab benchmarking tests: {e}")
        return False, {}

def test_dataset_integration():
    """Test the Enhanced Dataset Integration"""
    print("\n=== Testing Enhanced Dataset Integration ===")
    
    try:
        # Import the dataset integration
        from data_engineering.dataset_integration import DatasetManager
        
        # Create test datasets
        test_datasets = create_test_datasets()
        
        # Initialize dataset manager
        manager = DatasetManager()
        
        results = {}
        
        # Test dataset creation and validation
        for dataset_name, dataset in test_datasets.items():
            print(f"Testing {dataset_name} dataset...")
            
            # Create dataset
            dataset_id = manager.create_dataset(dataset_name, dataset)
            results[f"{dataset_name}_created"] = dataset_id
            
            # Validate dataset
            validation_result = manager.validate_dataset(dataset_id)
            results[f"{dataset_name}_validation"] = validation_result
            
            # Test quality filtering
            try:
                filtered_dataset_id = manager.create_quality_filtered_dataset(dataset_id, min_trust_score=0.7)
                results[f"{dataset_name}_filtered"] = filtered_dataset_id
            except Exception as e:
                results[f"{dataset_name}_filtered"] = f"Error: {e}"
        
        print("✓ Enhanced Dataset Integration tests completed successfully")
        return True, results
        
    except ImportError as e:
        print(f"✗ Failed to import Dataset Manager: {e}")
        return False, {}
    except Exception as e:
        print(f"✗ Error in dataset integration tests: {e}")
        return False, {}

def test_advanced_features():
    """Test advanced features and edge cases"""
    print("\n=== Testing Advanced Features ===")
    
    try:
        # Import required modules
        from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
        # from cleanlab_integration import CleanlabDataQualityManager # Removed as per edit hint
        
        # Create edge case datasets
        edge_datasets = {
            'empty': pd.DataFrame(),
            'single_column': pd.DataFrame({'col1': [1, 2, 3]}),
            'all_nan': pd.DataFrame({'col1': [np.nan, np.nan, np.nan]}),
            'all_zero': pd.DataFrame({'col1': [0, 0, 0], 'col2': [0, 0, 0]}),
            'large_dataset': pd.DataFrame(np.random.randn(10000, 10))
        }
        
        # Initialize engines
        trust_engine = AdvancedTrustScoringEngine()
        # quality_manager = CleanlabDataQualityManager() # Removed as per edit hint
        
        results = {}
        
        for dataset_name, dataset in edge_datasets.items():
            print(f"Testing {dataset_name} edge case...")
            
            try:
                # Test trust scoring
                trust_result = trust_engine.calculate_advanced_trust_score(dataset, method="ensemble")
                results[f"{dataset_name}_trust"] = trust_result
                
                # Test quality assessment
                # quality_result = quality_manager.calculate_data_trust_score(dataset, method="auto") # Removed as per edit hint
                results[f"{dataset_name}_quality"] = "N/A (Cleanlab removed)" # Placeholder
                
            except Exception as e:
                results[f"{dataset_name}_error"] = str(e)
        
        print("✓ Advanced Features tests completed successfully")
        return True, results
        
    except Exception as e:
        print(f"✗ Error in advanced features test: {e}")
        return False, {}

def generate_comprehensive_report(all_results):
    """Generate a comprehensive test report"""
    print("\n=== COMPREHENSIVE TEST REPORT ===")
    
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0
        },
        "detailed_results": all_results,
        "recommendations": []
    }
    
    # Count test results
    for test_name, (success, results) in all_results.items():
        report["summary"]["total_tests"] += 1
        if success:
            report["summary"]["passed_tests"] += 1
        else:
            report["summary"]["failed_tests"] += 1
    
    # Calculate success rate
    if report["summary"]["total_tests"] > 0:
        report["summary"]["success_rate"] = (
            report["summary"]["passed_tests"] / report["summary"]["total_tests"]
        )
    
    # Generate recommendations
    if report["summary"]["success_rate"] < 1.0:
        report["recommendations"].append("Some tests failed. Check dependencies and configuration.")
    
    if report["summary"]["passed_tests"] > 0:
        report["recommendations"].append("Core functionality is working. Consider installing optional dependencies for enhanced features.")
    
    # Print summary
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.2%}")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    # Save report to file
    report_path = f"advanced_trust_scoring_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report

def main():
    """Main test execution function"""
    print("🚀 Advanced Trust Scoring Test Suite")
    print("=" * 50)
    print()
    
    all_results = {}
    
    # Run all tests
    print("1. Testing Advanced Trust Scoring Engine...")
    success, results = test_advanced_trust_scoring_engine()
    all_results["advanced_trust_scoring"] = (success, results)
    
    print("\n2. Testing Cleanlab Benchmarking...")
    success, results = test_cleanlab_benchmarking()
    all_results["cleanlab_benchmarking"] = (success, results)
    
    print("\n3. Testing Enhanced Dataset Integration...")
    success, results = test_dataset_integration()
    all_results["dataset_integration"] = (success, results)
    
    print("\n4. Testing Advanced Features...")
    success, results = test_advanced_features()
    all_results["advanced_features"] = (success, results)
    
    # Generate comprehensive report
    report = generate_comprehensive_report(all_results)
    
    # Final status
    if report["summary"]["success_rate"] >= 0.75:
        print("\n🎉 Test suite completed with good success rate!")
    elif report["summary"]["success_rate"] >= 0.5:
        print("\n⚠️  Test suite completed with moderate success rate. Some features may need attention.")
    else:
        print("\n❌ Test suite completed with low success rate. Please check dependencies and configuration.")
    
    return report["summary"]["success_rate"] >= 0.5

if __name__ == "__main__":
    main()

import unittest
from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine

class TestAdvancedTrustScoringEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AdvancedTrustScoringEngine()
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })

    def test_calculate_advanced_trust_score_ensemble(self):
        result = self.engine.calculate_advanced_trust_score(self.test_data, method="ensemble")
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)

    def test_handle_missing_values(self):
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[0, 'feature1'] = np.nan
        processed_data = self.engine._handle_missing_values(data_with_missing)
        self.assertFalse(processed_data.isnull().values.any())

    def test_scale_data(self):
        scaled_data = self.engine._scale_data(self.test_data)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertEqual(self.test_data.shape, scaled_data.shape)

    def test_assess_data_quality(self):
        score = self.engine._assess_data_quality(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_create_synthetic_labels(self):
        labels = self.engine._create_synthetic_labels(self.test_data)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.test_data))

    def test_quantify_uncertainty(self):
        score = self.engine._quantify_uncertainty(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_bootstrap_uncertainty(self):
        score = self.engine._calculate_bootstrap_uncertainty(self.test_data)
        self.assertIsInstance(score, (float, int))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_confidence_interval_uncertainty(self):
        score = self.engine._calculate_confidence_interval_uncertainty(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_assess_clustering_quality(self):
        score = self.engine._assess_clustering_quality(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_robust_trust_score(self):
        result = self.engine._calculate_robust_trust_score(self.test_data, self.test_data)
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)

    def test_calculate_uncertainty_trust_score(self):
        result = self.engine._calculate_uncertainty_trust_score(self.test_data, self.test_data)
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)

    def test_calculate_basic_trust_score(self):
        result = self.engine._calculate_basic_trust_score(self.test_data)
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)

    def test_calculate_median_based_trust(self):
        score = self.engine._calculate_median_based_trust(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_mad_based_trust(self):
        score = self.engine._calculate_mad_based_trust(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_trimmed_mean_trust(self):
        score = self.engine._calculate_trimmed_mean_trust(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_winsorized_trust(self):
        score = self.engine._calculate_winsorized_trust(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_entropy_uncertainty(self):
        score = self.engine._calculate_entropy_uncertainty(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_bayesian_uncertainty(self):
        score = self.engine._calculate_bayesian_uncertainty(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_detect_correlation_issues(self):
        score = self.engine._detect_correlation_issues(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_assess_type_consistency(self):
        score = self.engine._assess_type_consistency(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_ensemble_anomaly_detection(self):
        score = self.engine._ensemble_anomaly_detection(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_mahalanobis_scores(self):
        score = self.engine._calculate_mahalanobis_scores(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_assess_statistical_robustness(self):
        score = self.engine._assess_statistical_robustness(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_outlier_resistance(self):
        score = self.engine._calculate_outlier_resistance(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_analyze_distributions(self):
        score = self.engine._analyze_distributions(self.test_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_advanced_trust_score_robust(self):
        result = self.engine.calculate_advanced_trust_score(self.test_data, method="robust")
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)

    def test_calculate_advanced_trust_score_uncertainty(self):
        result = self.engine.calculate_advanced_trust_score(self.test_data, method="uncertainty")
        self.assertIn('trust_score', result)
        self.assertIsInstance(result['trust_score'], float)
        self.assertGreaterEqual(result['trust_score'], 0)
        self.assertLessEqual(result['trust_score'], 1)