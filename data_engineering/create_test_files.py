#!/usr/bin/env python3
"""
Create test files in multiple formats for testing upload functionality
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path

def create_test_files():
    """Create test files in various formats"""
    print("📁 Creating test files in multiple formats...")
    
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'feature3': np.random.normal(10, 3, 100),
        'feature4': np.random.randint(0, 10, 100),
        'feature5': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Create test files directory
    test_dir = Path("./test_files")
    test_dir.mkdir(exist_ok=True)
    
    # 1. CSV file
    csv_path = test_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Created: {csv_path}")
    
    # 2. JSON file
    json_path = test_dir / "test_data.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"✅ Created: {json_path}")
    
    # 3. Excel file
    excel_path = test_dir / "test_data.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"✅ Created: {excel_path}")
    
    # 4. Parquet file
    parquet_path = test_dir / "test_data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Created: {parquet_path}")
    
    # 5. Pickle file
    pickle_path = test_dir / "test_data.pkl"
    df.to_pickle(pickle_path)
    print(f"✅ Created: {pickle_path}")
    
    # 6. Feather file
    feather_path = test_dir / "test_data.feather"
    df.to_feather(feather_path)
    print(f"✅ Created: {feather_path}")
    
    # 7. YAML file (structured data)
    yaml_data = {
        'dataset_info': {
            'name': 'test_dataset',
            'rows': len(df),
            'columns': len(df.columns),
            'features': list(df.columns)
        },
        'sample_data': df.head(5).to_dict('records')
    }
    
    yaml_path = test_dir / "test_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"✅ Created: {yaml_path}")
    
    # 8. XML file
    xml_path = test_dir / "test_data.xml"
    df.to_xml(xml_path, index=False)
    print(f"✅ Created: {xml_path}")
    
    print(f"\n🎉 Created {len(list(test_dir.glob('*')))} test files in {test_dir}")
    print("📋 Available test files:")
    for file_path in test_dir.glob("*"):
        file_size = file_path.stat().st_size / 1024
        print(f"   - {file_path.name} ({file_size:.1f} KB)")
    
    return test_dir

def test_file_loading():
    """Test loading all created files"""
    print("\n🧪 Testing file loading...")
    
    from data_engineering.trust_scoring_dashboard import TrustScoringDashboard
    dashboard = TrustScoringDashboard()
    
    test_dir = Path("./test_files")
    if not test_dir.exists():
        print("❌ Test files directory not found")
        return
    
    for file_path in test_dir.glob("*"):
        print(f"\n📄 Testing: {file_path.name}")
        
        # Validate format
        is_valid, format_type = dashboard.validate_file_format(str(file_path))
        print(f"   Format validation: {is_valid}, Type: {format_type}")
        
        if is_valid:
            # Try to load data
            try:
                df_loaded = dashboard.load_data_by_format(str(file_path), format_type)
                if df_loaded is not None:
                    print(f"   ✅ Loaded: {len(df_loaded)} rows, {len(df_loaded.columns)} columns")
                else:
                    print(f"   ❌ Failed to load data")
            except Exception as e:
                print(f"   ❌ Error loading: {e}")
        else:
            print(f"   ⚠️  Skipped due to format validation failure")

if __name__ == "__main__":
    print("🚀 Test File Creation and Validation")
    print("=" * 50)
    
    # Create test files
    create_test_files()
    
    # Test file loading
    test_file_loading()
    
    print("\n🎯 Ready for dashboard testing!")
    print("💡 You can now use these files to test the upload functionality in the Streamlit dashboard.") 