# Project 1: Building Data Model and Writing ETL Job

Reference: https://lnkd.in/dQSm-d_k

## Instructions
1. Download the dataset from Kaggle as described in the project link above.
2. Place the raw data file (e.g., `raw_data.csv`) in the `data/` subfolder of this project.
3. Install dependencies:
   ```bash
   pip install pandas
   # (Optional for real data quality checks)
   pip install great_expectations
   ```
4. Run the ETL pipeline:
   ```bash
   python3 project_etl.py
   ```

## What this does
- Loads the raw data
- Infers a data model/schema
- Simulates ETL pipeline using modular connectors
- Runs a basic data quality check (no nulls)
- Simulates dashboard/report generation
- Prints governance logs and metrics

## Customization
- Replace the stub connectors with real logic as needed
- Integrate Great Expectations or other data quality tools for advanced checks
- Extend the pipeline for your use case
