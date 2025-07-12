#!/usr/bin/env python3
"""
Test script for Cleanlab Integration
Demonstrates confident learning and data quality assessment features
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_integration import DatasetManager
from cleanlab_integration import CleanlabDataQualityManager

def create_test_dataset_with_issues():
    """Create a test dataset with various quality issues"""
    np.random.seed(42)
    
    # Create base data
    n_samples = 1000
    data = pd.DataFrame({
        'id': range(1, n_samples + 1),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Introduce quality issues
    # 1. Missing values
    data.loc[100:150, 'feature1'] = np.nan
    
    # 2. Duplicate rows
    data.loc[200:220] = data.loc[100:120].values
    
    # 3. Outliers
    data.loc[300:310, 'feature2'] = 1000  # Extreme outliers
    
    # 4. Zero variance features
    data.loc[400:500, 'feature3'] = 0
    
    # 5. Label noise (for confident learning)
    noisy_labels = data['target'].copy()
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]  # Flip labels
    
    return data, noisy_labels.tolist()

def test_cleanlab_data_quality_manager():
    """Test the CleanlabDataQualityManager directly"""
    print("=== Testing CleanlabDataQualityManager ===")
    
    # Create test data
    data, labels = create_test_dataset_with_issues()
    
    # Initialize manager
    manager = CleanlabDataQualityManager()
    
    # Test trust score calculation
    print("\n1. Calculating Data Trust Score...")
    trust_result = manager.calculate_data_trust_score(data, labels)
    
    if 'error' not in trust_result:
        print(f"Trust Score: {trust_result.get('trust_score', 0):.3f}")
        print(f"Label Issues Count: {trust_result.get('label_issues_count', 0)}")
        print(f"Quality Distribution: {trust_result.get('quality_distribution', {})}")
    else:
        print(f"Error: {trust_result['error']}")
    
    # Test automated validation
    print("\n2. Running Automated Validation...")
    validation_result = manager.automated_data_validation(data)
    
    if 'error' not in validation_result:
        print(f"Validation Passed: {validation_result.get('validation_passed', False)}")
        print(f"Issues Found: {len(validation_result.get('issues_found', []))}")
        for issue in validation_result.get('issues_found', []):
            print(f"  - {issue}")
    else:
        print(f"Error: {validation_result['error']}")
    
    # Test quality-based filtering
    print("\n3. Testing Quality-Based Filtering...")
    filtered_data = manager.create_quality_based_filter(data, min_trust_score=0.7)
    print(f"Original rows: {len(data)}")
    print(f"Filtered rows: {len(filtered_data)}")
    print(f"Retention rate: {len(filtered_data)/len(data)*100:.1f}%")
    
    # Test quality report generation
    print("\n4. Generating Quality Report...")
    report = manager.generate_quality_report(data)
    print("Quality report generated successfully")
    
    return manager

def test_dataset_manager_integration():
    """Test Cleanlab integration with DatasetManager"""
    print("\n=== Testing DatasetManager Integration ===")
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Create test data
    data, labels = create_test_dataset_with_issues()
    
    # Create dataset
    print("1. Creating dataset...")
    dataset_id = dataset_manager.create_dataset('test_cleanlab', data)
    print(f"Dataset created: {dataset_id}")
    
    # Test standard validation
    print("\n2. Running standard validation...")
    validation_results = dataset_manager.validate_dataset(dataset_id)
    print(f"Standard validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    # Test Cleanlab validation
    if dataset_manager.cleanlab_manager:
        print("\n3. Running Cleanlab validation...")
        cleanlab_results = dataset_manager.validate_dataset_with_cleanlab(dataset_id, labels)
        
        if 'error' not in cleanlab_results:
            trust_score = cleanlab_results.get('trust_assessment', {}).get('trust_score', 0)
            print(f"Cleanlab trust score: {trust_score:.3f}")
            
            # Test quality filtering
            print("\n4. Creating quality-filtered dataset...")
            filtered_dataset_id = dataset_manager.create_quality_filtered_dataset(
                dataset_id, min_trust_score=0.7
            )
            print(f"Quality-filtered dataset: {filtered_dataset_id}")
            
            # Compare datasets
            original_df = dataset_manager.load_dataset(dataset_id)
            filtered_df = dataset_manager.load_dataset(filtered_dataset_id)
            print(f"Original dataset: {len(original_df)} rows")
            print(f"Filtered dataset: {len(filtered_df)} rows")
            print(f"Quality filtering removed: {len(original_df) - len(filtered_df)} rows")
        else:
            print(f"Cleanlab error: {cleanlab_results['error']}")
    else:
        print("Cleanlab not available in DatasetManager")
    
    return dataset_manager

def test_advanced_features():
    """Test advanced Cleanlab features"""
    print("\n=== Testing Advanced Features ===")
    
    # Create synthetic data with known patterns
    np.random.seed(42)
    n_samples = 500
    
    # Create data with different quality levels
    high_quality_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    low_quality_data = pd.DataFrame({
        'feature1': np.random.normal(0, 5, n_samples),  # High variance
        'feature2': np.random.choice([0, 1], n_samples),  # Binary feature
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Imbalanced
    })
    
    # Add noise to low quality data
    low_quality_data.loc[100:200, 'feature1'] = np.nan
    low_quality_data.loc[300:350, 'feature2'] = 999  # Outliers
    
    manager = CleanlabDataQualityManager()
    
    print("1. Testing high-quality data...")
    high_quality_result = manager.calculate_data_trust_score(high_quality_data)
    if 'error' not in high_quality_result:
        print(f"High quality trust score: {high_quality_result.get('trust_score', 0):.3f}")
    
    print("\n2. Testing low-quality data...")
    low_quality_result = manager.calculate_data_trust_score(low_quality_data)
    if 'error' not in low_quality_result:
        print(f"Low quality trust score: {low_quality_result.get('trust_score', 0):.3f}")
    
    print("\n3. Testing custom validation rules...")
    def custom_rule_1(df):
        """Check if feature1 has reasonable variance"""
        return df['feature1'].var() < 10
    
    def custom_rule_2(df):
        """Check if target is not too imbalanced"""
        target_counts = df['target'].value_counts()
        return min(target_counts) / max(target_counts) > 0.3
    
    validation_rules = {
        'reasonable_variance': custom_rule_1,
        'balanced_target': custom_rule_2
    }
    
    high_quality_validation = manager.automated_data_validation(high_quality_data, validation_rules)
    low_quality_validation = manager.automated_data_validation(low_quality_data, validation_rules)
    
    print(f"High quality validation passed: {high_quality_validation.get('validation_passed', False)}")
    print(f"Low quality validation passed: {low_quality_validation.get('validation_passed', False)}")

def main():
    """Run all tests"""
    print("🧪 Testing Cleanlab Integration for OpenTrustEval")
    print("=" * 60)
    
    try:
        # Test 1: Direct Cleanlab manager
        test_cleanlab_data_quality_manager()
        
        # Test 2: Dataset manager integration
        test_dataset_manager_integration()
        
        # Test 3: Advanced features
        test_advanced_features()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  - Confident learning for data quality assessment")
        print("  - Data trust scoring")
        print("  - Quality-based data filtering")
        print("  - Automated data validation")
        print("  - Integration with dataset management")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 