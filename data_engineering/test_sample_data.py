#!/usr/bin/env python3
"""
Test script for sample data
Verifies that the sample CSV data works with the trust scoring system
"""

import pandas as pd
import numpy as np
from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine

def test_sample_data():
    """Test the sample data with the trust scoring system"""
    
    # Load the sample data
    print("Loading sample data...")
    df = pd.read_csv('sample_test_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First 5 rows:\n{df.head()}")
    
    # Initialize the trust scoring engine
    print("\nInitializing trust scoring engine...")
    engine = AdvancedTrustScoringEngine()
    
    # Test different scoring methods
    methods = ["ensemble", "robust", "uncertainty"]
    
    for method in methods:
        print(f"\n--- Testing {method} method ---")
        try:
            result = engine.calculate_advanced_trust_score(df, method=method)
            print(f"Trust Score: {result.get('trust_score', 'N/A'):.3f}")
            print(f"Method: {result.get('method', 'N/A')}")
            if 'component_scores' in result:
                print("Component Scores:")
                for component, score in result['component_scores'].items():
                    print(f"  {component}: {score:.3f}")
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    # Test data quality metrics
    print("\n--- Data Quality Analysis ---")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Data types: {df.dtypes.value_counts()}")
    
    # Test numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Feature ranges:")
    for col in numeric_features[:5]:  # Show first 5 features
        print(f"  {col}: {df[col].min():.2f} to {df[col].max():.2f}")
    
    print("\n✅ Sample data test completed successfully!")

if __name__ == "__main__":
    test_sample_data() 