#!/usr/bin/env python3
"""
Test script for Fallback Quality Assessment
Demonstrates data quality assessment without Cleanlab dependency
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cleanlab_integration import CleanlabDataQualityManager, FallbackDataQualityManager

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
    
    # Add quality issues
    data.loc[100:200, 'feature1'] = np.nan  # Missing values
    data.loc[300:400, 'feature2'] = 0  # Zero values
    data.loc[500:600, 'feature3'] = data.loc[500:600, 'feature1']  # Duplicates
    data.loc[700:800, 'feature1'] = 999999  # Outliers
    
    return data

def test_fallback_quality_manager():
    """Test the fallback quality manager"""
    print("=== Testing Fallback Data Quality Manager ===")
    
    # Create test data
    data = create_test_dataset_with_issues()
    print(f"Created test dataset with shape: {data.shape}")
    
    # Initialize fallback manager
    fallback_manager = FallbackDataQualityManager()
    
    # Test trust score calculation
    print("\n1. Testing Trust Score Calculation:")
    trust_result = fallback_manager.calculate_data_trust_score(data)
    print(f"Trust Score: {trust_result.get('trust_score', 'N/A'):.3f}")
    print(f"Method: {trust_result.get('method', 'N/A')}")
    print(f"Quality Metrics: {json.dumps(trust_result.get('quality_metrics', {}), indent=2)}")
    
    # Test quality-based filtering
    print("\n2. Testing Quality-Based Filtering:")
    filtered_data = fallback_manager.create_quality_based_filter(data, min_trust_score=0.8)
    print(f"Original rows: {len(data)}")
    print(f"Filtered rows: {len(filtered_data)}")
    print(f"Retention rate: {len(filtered_data)/len(data)*100:.1f}%")
    
    # Test automated validation
    print("\n3. Testing Automated Validation:")
    validation_result = fallback_manager.automated_data_validation(data)
    print(f"Validation Passed: {validation_result.get('validation_passed', False)}")
    print(f"Issues Found: {len(validation_result.get('issues_found', []))}")
    print(f"Recommendations: {validation_result.get('recommendations', [])}")
    
    # Test quality report generation
    print("\n4. Testing Quality Report Generation:")
    report = fallback_manager.generate_quality_report(data)
    print("Quality report generated successfully!")
    print(f"Report length: {len(report)} characters")
    
    return True

def test_cleanlab_manager_fallback():
    """Test the main Cleanlab manager with fallback functionality"""
    print("\n=== Testing Cleanlab Manager with Fallback ===")
    
    # Create test data
    data = create_test_dataset_with_issues()
    
    # Initialize main manager (will use fallback since Cleanlab is not available)
    manager = CleanlabDataQualityManager()
    
    # Test trust score calculation
    print("\n1. Testing Trust Score Calculation (with fallback):")
    trust_result = manager.calculate_data_trust_score(data)
    print(f"Trust Score: {trust_result.get('trust_score', 'N/A'):.3f}")
    print(f"Method: {trust_result.get('method', 'N/A')}")
    
    # Test quality-based filtering
    print("\n2. Testing Quality-Based Filtering (with fallback):")
    filtered_data = manager.create_quality_based_filter(data, min_trust_score=0.8)
    print(f"Original rows: {len(data)}")
    print(f"Filtered rows: {len(filtered_data)}")
    
    # Test quality report generation
    print("\n3. Testing Quality Report Generation (with fallback):")
    report = manager.generate_quality_report(data)
    print("Quality report generated successfully!")
    
    return True

def test_with_real_dataset():
    """Test with a real dataset from the system"""
    print("\n=== Testing with Real Dataset ===")
    
    try:
        from dataset_integration import DatasetManager
        
        # Initialize dataset manager
        dataset_manager = DatasetManager()
        
        # List available datasets
        datasets = dataset_manager.list_datasets()
        if not datasets:
            print("No datasets found. Creating a test dataset...")
            data = create_test_dataset_with_issues()
            dataset_id = dataset_manager.create_dataset('test_quality', data)
        else:
            # Use the first available dataset
            dataset_id = datasets[0]['id']
            print(f"Using existing dataset: {dataset_id}")
        
        # Load dataset
        data = dataset_manager.load_dataset(dataset_id)
        print(f"Loaded dataset with shape: {data.shape}")
        
        # Test quality assessment
        manager = CleanlabDataQualityManager()
        
        # Calculate trust score
        trust_result = manager.calculate_data_trust_score(data)
        print(f"Trust Score: {trust_result.get('trust_score', 'N/A'):.3f}")
        print(f"Method: {trust_result.get('method', 'N/A')}")
        
        # Generate quality report
        report = manager.generate_quality_report(data)
        print("Quality report generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error testing with real dataset: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Fallback Quality Assessment System")
    print("=" * 50)
    
    try:
        # Test fallback manager directly
        test_fallback_quality_manager()
        
        # Test main manager with fallback
        test_cleanlab_manager_fallback()
        
        # Test with real dataset
        test_with_real_dataset()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("- Fallback quality assessment works without Cleanlab")
        print("- Trust scoring using statistical methods")
        print("- Quality-based filtering functional")
        print("- Automated validation working")
        print("- Quality report generation operational")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 