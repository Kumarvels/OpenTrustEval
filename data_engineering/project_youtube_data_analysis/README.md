# Project 3: YouTube Data Analysis (End-To-End Data Engineering Project)

Reference: https://lnkd.in/dS8FwupT

## Instructions
1. Download the YouTube dataset from Kaggle as described in the project link above.
2. Place the raw data file (e.g., `youtube_data.csv`) in the `data/` subfolder of this project.
3. Install dependencies:
   ```bash
   pip install pandas
   # (Optional for real data quality checks)
   pip install great_expectations
   ```
4. Run the ETL pipeline:
   ```bash
   python3 project_youtube_etl.py
   ```

## What this does
- Loads the YouTube data
- Infers a data model/schema
- Simulates ETL pipeline using modular connectors
- Runs a basic data quality check (no nulls)
- Simulates dashboard/report generation
- Prints governance logs and metrics

## Customization
- Replace the stub connectors with real logic as needed
- Integrate Great Expectations or other data quality tools for advanced checks
- Extend the pipeline for your use case
