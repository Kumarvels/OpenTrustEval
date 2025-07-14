#!/usr/bin/env python3
"""
Custom Cleanlab Data Issue Detection Pipeline (CLI)
- Ingest CSV
- Run Cleanlab Datalab plugin
- Filter out low-confidence rows (optional)
- Save cleaned CSV and issues JSON
"""
import argparse
import pandas as pd
import json
import sys
from plugins import cleanlab_datalab_plugin

def main():
    parser = argparse.ArgumentParser(description="Custom Cleanlab Data Issue Detection Pipeline")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output-cleaned', required=True, help='Output cleaned CSV file')
    parser.add_argument('--output-issues', required=True, help='Output issues JSON file')
    parser.add_argument('--label-column', default='label', help='Label column for Cleanlab')
    parser.add_argument('--confidence-threshold', type=float, default=None, help='Confidence threshold for issues')
    parser.add_argument('--issue-types', type=str, default=None, help='Comma-separated list of issue types to include')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    labels = df[args.label_column].tolist() if args.label_column in df.columns else None
    config = {
        'enable_cleanlab': True,
        'cleanlab_label_column': args.label_column,
    }
    if args.confidence_threshold is not None:
        config['cleanlab_confidence_threshold'] = args.confidence_threshold
    if args.issue_types:
        config['cleanlab_issue_types'] = [t.strip() for t in args.issue_types.split(',')]

    print(f"Running Cleanlab plugin with config: {config}")
    result = cleanlab_datalab_plugin.cleanlab_datalab_plugin(df, labels=labels, config=config)
    print(f"Detected {len(result['issues']) if result['issues'] else 0} issues.")
    with open(args.output_issues, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Issues saved to {args.output_issues}")
    # Optionally filter out rows with issues (e.g., label errors or low confidence)
    if result['issues']:
        issue_rows = set(iss['row'] for iss in result['issues'] if 'row' in iss)
        cleaned_df = df[~df.index.isin(issue_rows)]
    else:
        cleaned_df = df
    cleaned_df.to_csv(args.output_cleaned, index=False)
    print(f"Cleaned data saved to {args.output_cleaned}")

if __name__ == "__main__":
    main() 