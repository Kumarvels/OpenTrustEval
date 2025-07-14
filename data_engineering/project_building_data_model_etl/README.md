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

## Cleanlab Datalab Plugin Integration (Optional Data Issue Detection)

This pipeline supports optional data issue detection using the Cleanlab Datalab plugin.

### How to Enable
- By default, Cleanlab is **disabled**.
- To enable, run:
  ```sh
  python project_etl.py --enable-cleanlab
  ```
- Or, in code, set `config = {'enable_cleanlab': True}` when creating `DataLifecycleManager`.

### Requirements
- Cleanlab must be installed (`pip install cleanlab`).
- You must have a valid Cleanlab Datalab license and user consent for commercial use. See [Cleanlab Pricing](https://cleanlab.ai/pricing/).

### What Happens
- After loading the raw data, the pipeline will run the Cleanlab plugin if enabled.
- The plugin will print detected data issues and warnings to the console.
- If Cleanlab is not installed or not enabled, a warning is shown and the pipeline continues.

### Example Output
```
[Cleanlab Datalab Plugin Output]
{'plugin_output': 'Detected 3 data issues via Cleanlab Datalab.', 'issues': [...], 'warnings': ['⚠️ Ensure you have a valid Cleanlab Datalab license and user consent before using this plugin.']}
```

### Troubleshooting
- If you see `Cleanlab is not installed. Plugin is inactive.`, install Cleanlab and try again.
- If you see commercial/consent warnings, ensure you have the proper license and user permissions.
- If you see a `UnicodeEncodeError` or missing plugin output in Windows, set your terminal encoding to UTF-8 (e.g., run `chcp 65001` in CMD/PowerShell) or redirect output to a UTF-8 file.
- The pipeline works normally even if the plugin is not used.

### References
- [Cleanlab Datalab Plugin README](../../plugins/README_CLEANLAB_PLUGIN.md)
- [Cleanlab Documentation](https://cleanlab.ai/docs/datalab/)

### Advanced Options
- **Label column selection:**
  - Use `--cleanlab-label-column job_title` to specify which column to use as labels for Cleanlab analysis.
- **Save output to file:**
  - Use `--save-cleanlab-output` to save plugin output to `cleanlab_issues.json` for review.

#### Example CLI Usage
```sh
python project_etl.py --enable-cleanlab --cleanlab-label-column job_title --save-cleanlab-output
```
