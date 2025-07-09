"""
Synthetic Covid-19 Data Generator

Purpose:
    Generate a synthetic Covid-19 dataset matching the typical structure of covid_19_data.csv for testing and demonstration.
    This script is useful for robust pipeline testing when real data is unavailable or for privacy reasons.

How it helps:
    - Enables end-to-end pipeline validation without real data.
    - Ensures data quality checks and dashboards can be tested.
    - Can be parameterized for different dataset sizes or distributions.

Usage:
    python3 generate_synthetic_covid_data.py
    # Or modify n_rows for a different dataset size.

Columns:
    SNo, ObservationDate, Province/State, Country/Region, Last Update, Confirmed, Deaths, Recovered
"""
import pandas as pd
import numpy as np
import os

# Define columns based on typical covid_19_data.csv structure
columns = [
    'SNo', 'ObservationDate', 'Province/State', 'Country/Region', 'Last Update',
    'Confirmed', 'Deaths', 'Recovered'
]

n_rows = 10000
np.random.seed(42)

data = {
    'SNo': np.arange(1, n_rows + 1),
    'ObservationDate': pd.date_range('2020-01-22', periods=n_rows, freq='H').strftime('%m/%d/%Y'),
    'Province/State': np.random.choice(['Hubei', 'Guangdong', 'New York', 'California', 'Unknown'], n_rows),
    'Country/Region': np.random.choice(['China', 'US', 'Italy', 'Spain', 'India'], n_rows),
    'Last Update': pd.date_range('2020-01-22', periods=n_rows, freq='H').strftime('%m/%d/%Y %H:%M'),
    'Confirmed': np.random.randint(0, 100000, n_rows),
    'Deaths': np.random.randint(0, 5000, n_rows),
    'Recovered': np.random.randint(0, 90000, n_rows)
}

df = pd.DataFrame(data)

os.makedirs('data_engineering/project_covid_data_analysis/data', exist_ok=True)
df.to_csv('data_engineering/project_covid_data_analysis/data/covid_19_data.csv', index=False)
print('Synthetic covid_19_data.csv created with 10,000 rows.')
