"""
Project 2: Covid Data Analysis Project

- Reference: https://lnkd.in/d2u4tS3R
- Dataset: (Auto-download from Kaggle if not present)
- Demonstrates: End-to-end Covid-19 data engineering, ETL, data quality checks, dashboarding

Instructions:
1. Install dependencies:
   pip install pandas kaggle
2. Set up your Kaggle API token (see README).
3. Run this script to execute the ETL pipeline, data quality checks, and dashboard/report generation.

To run:
$ python3 data_engineering/project_covid_data_analysis/project_covid_etl.py
"""
import sys
import os
import pandas as pd
import subprocess
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_engineering.data_lifecycle import DataLifecycleManager, SparkConnector, DBTConnector, SnowflakeConnector, PowerBIConnector
from data_engineering.common_data_loader import resolve_data_path, robust_read_csv

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'covid_19_data.csv')
KAGGLE_DATASET = 'sudalairajkumar/novel-corona-virus-2019-dataset'
KAGGLE_FILE = 'covid_19_data.csv'


def download_covid_data():
    """Download Covid-19 data from Kaggle if not present."""
    if os.path.exists(DATA_PATH):
        print("Covid-19 dataset already present.")
        return
    print("Downloading Covid-19 dataset from Kaggle...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-f', KAGGLE_FILE, '-p', os.path.dirname(DATA_PATH), '--force'
        ], check=True)
        # Unzip if needed
        zip_path = os.path.join(os.path.dirname(DATA_PATH), KAGGLE_FILE + '.zip')
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(DATA_PATH))
            os.remove(zip_path)
    except Exception as e:
        print(f"Kaggle download failed: {e}\nPlease download manually and place in the 'data/' folder.")
        sys.exit(1)

def build_covid_data_model(df):
    """Build a data model/schema for Covid-19 data."""
    return df.dtypes.to_dict()

def covid_etl_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-path', type=str, help='Path to local Covid CSV file')
    parser.add_argument('--gdrive-path', type=str, help='Path to Covid CSV file on Google Drive (Colab only)')
    parser.add_argument('--enable-cleanlab', action='store_true', help='Enable Cleanlab Datalab plugin for data issue detection')
    parser.add_argument('--cleanlab-label-column', type=str, help='Column to use as label for Cleanlab plugin')
    parser.add_argument('--save-cleanlab-output', action='store_true', help='Save Cleanlab plugin output to cleanlab_issues.json')
    args, _ = parser.parse_known_args()
    download_covid_data()
    config = {
        'enable_cleanlab': args.enable_cleanlab,
        'cleanlab_label_column': args.cleanlab_label_column
    }
    manager = DataLifecycleManager(config=config)
    # Step 1: Load raw data
    data_path = resolve_data_path(DATA_PATH, args)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}\nPlease download or upload the file.")
        sys.exit(1)
    df = robust_read_csv(data_path)
    print("Loaded Covid-19 data:", df.head())
    # Step 1.5: Run Cleanlab plugin if enabled
    label_col = config.get('cleanlab_label_column')
    labels = df[label_col].tolist() if label_col and label_col in df.columns else None
    cleanlab_result = manager.run_cleanlab_plugin(df, labels=labels)
    if cleanlab_result:
        print("\n[Cleanlab Datalab Plugin Output]")
        print(cleanlab_result)
        if args.save_cleanlab_output:
            import json
            with open('cleanlab_issues.json', 'w', encoding='utf-8') as f:
                json.dump(cleanlab_result, f, ensure_ascii=False, indent=2)
                print("Cleanlab plugin output saved to cleanlab_issues.json")
    # Step 2: Build data model
    schema = build_covid_data_model(df)
    print("Inferred schema:", schema)
    # Step 3: Synthetic data (optional)
    manager.generate_synthetic_data({'rows': 100, 'schema': schema})
    # Step 4: Upload data (simulate)
    manager.upload_data('file', DATA_PATH)
    # Step 5: Load to DB (simulate)
    manager.add_db('snowflake', SnowflakeConnector('account', 'user', 'pass', 'db'))
    # Step 6: ELT pipeline (simulate)
    manager.add_connector('spark', SparkConnector({'master': 'local'}))
    manager.add_connector('dbt', DBTConnector('/my/dbt/project'))
    elt_pipeline = [
        {'tool': 'spark', 'action': 'run_job', 'args': {'job': 'covid_elt_job'}},
        {'tool': 'dbt', 'action': 'run', 'args': {'model': 'covid_clean_model'}},
    ]
    manager.run_pipeline(elt_pipeline)
    # Step 7: Data quality check (stub)
    print("Running data quality checks...")
    assert not df.isnull().any().any(), "Nulls found in Covid-19 data!"
    print("Data quality check passed!")
    # Step 8: Dashboard/report (simulate)
    manager.add_connector('powerbi', PowerBIConnector('workspace', 'token'))
    manager.dashboard('powerbi', {'title': 'Covid-19 Data Report'})
    # Step 9: Print summary dashboard (console)
    print("\n=== Covid-19 Data Dashboard ===")
    print("Sample data:")
    print(df.head(10))
    print("Summary stats:")
    print(df.describe(include='all'))
    print('Governance logs:', manager.get_governance_logs())
    print('Metrics:', manager.get_metrics())
    # Step 10: End-to-end test assertions
    assert manager.metrics['synthetic_data_generated'] == 1
    assert manager.metrics['uploads'] == 1
    assert manager.metrics['elt_runs'] == 2
    print("End-to-end test passed!")

if __name__ == "__main__":
    covid_etl_pipeline()
