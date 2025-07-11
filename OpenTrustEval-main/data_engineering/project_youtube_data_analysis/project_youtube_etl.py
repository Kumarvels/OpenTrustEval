"""
Project 3: YouTube Data Analysis (End-To-End Data Engineering Project)

- Reference: https://lnkd.in/dS8FwupT
- Dataset: (Auto-download from Kaggle if not present)
- Demonstrates: End-to-end YouTube data engineering, ETL, data quality checks, dashboarding

Instructions:
1. Install dependencies:
   pip install pandas kaggle
2. Set up your Kaggle API token (see README).
3. Run this script to execute the ETL pipeline, data quality checks, and dashboard/report generation.

To run:
$ python3 data_engineering/project_youtube_data_analysis/project_youtube_etl.py
"""
import sys
import os
import pandas as pd
import subprocess
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_engineering.data_lifecycle import DataLifecycleManager, SparkConnector, DBTConnector, SnowflakeConnector, PowerBIConnector
from data_engineering.common_data_loader import resolve_data_path, robust_read_csv

try:
    from google.colab import drive as gdrive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'youtube_data.csv')
KAGGLE_DATASET = 'datasnaek/youtube-new'
KAGGLE_FILE = 'INvideos.csv'  # Example: India YouTube data


def download_youtube_data():
    """Download YouTube data from Kaggle if not present."""
    if os.path.exists(DATA_PATH):
        # Check if file is a ZIP masquerading as CSV (corrupted from previous runs)
        with open(DATA_PATH, 'rb') as f:
            sig = f.read(4)
        if sig == b'PK\x03\x04':
            print("Corrupted CSV detected (ZIP signature). Removing and redownloading...")
            os.remove(DATA_PATH)
        else:
            print("YouTube dataset already present.")
            return
    print("Downloading YouTube dataset from Kaggle...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-f', KAGGLE_FILE, '-p', os.path.dirname(DATA_PATH), '--force'
        ], check=True)
        # Handle Kaggle quirk: sometimes the file is a ZIP named as CSV
        kaggle_file_path = os.path.join(os.path.dirname(DATA_PATH), KAGGLE_FILE)
        with open(kaggle_file_path, 'rb') as f:
            sig = f.read(4)
        if sig == b'PK\x03\x04':
            print("Kaggle file is a ZIP masquerading as CSV. Extracting...")
            import zipfile
            with zipfile.ZipFile(kaggle_file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(DATA_PATH))
            os.remove(kaggle_file_path)
        # After extraction, ensure the CSV exists
        extracted_csv = os.path.join(os.path.dirname(DATA_PATH), KAGGLE_FILE.replace('.zip', ''))
        if os.path.exists(extracted_csv):
            os.rename(extracted_csv, DATA_PATH)
        else:
            # fallback: look for any CSV in the folder
            for fname in os.listdir(os.path.dirname(DATA_PATH)):
                if fname.endswith('.csv'):
                    os.rename(os.path.join(os.path.dirname(DATA_PATH), fname), DATA_PATH)
                    break
    except Exception as e:
        print(f"Kaggle download failed: {e}\nPlease download manually and place in the 'data/' folder.")
        sys.exit(1)

def build_youtube_data_model(df):
    """Build a data model/schema for YouTube data."""
    return df.dtypes.to_dict()

def youtube_etl_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-path', type=str, help='Path to local YouTube CSV file')
    parser.add_argument('--gdrive-path', type=str, help='Path to YouTube CSV file on Google Drive (Colab only)')
    args, _ = parser.parse_known_args()
    download_youtube_data()
    manager = DataLifecycleManager()
    # Step 1: Load raw data
    data_path = resolve_data_path(DATA_PATH, args)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}\nPlease download or upload the file.")
        sys.exit(1)
    df = robust_read_csv(data_path)
    print("Loaded YouTube data:", df.head())
    # Step 2: Build data model
    schema = build_youtube_data_model(df)
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
        {'tool': 'spark', 'action': 'run_job', 'args': {'job': 'youtube_elt_job'}},
        {'tool': 'dbt', 'action': 'run', 'args': {'model': 'youtube_clean_model'}},
    ]
    manager.run_pipeline(elt_pipeline)
    # Step 7: Data quality check (stub)
    print("Running data quality checks...")
    assert not df.isnull().any().any(), "Nulls found in YouTube data!"
    print("Data quality check passed!")
    # Step 8: Dashboard/report (simulate)
    manager.add_connector('powerbi', PowerBIConnector('workspace', 'token'))
    manager.dashboard('powerbi', {'title': 'YouTube Data Report'})
    # Step 9: Print summary dashboard (console)
    print("\n=== YouTube Data Dashboard ===")
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
    youtube_etl_pipeline()
