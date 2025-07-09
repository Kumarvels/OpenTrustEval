import os
import sys
import pandas as pd
import argparse

try:
    from google.colab import drive as gdrive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def resolve_data_path(default_path, args=None):
    """Resolve the data path based on user input or environment."""
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--local-path', type=str, help='Path to local CSV file')
        parser.add_argument('--gdrive-path', type=str, help='Path to CSV file on Google Drive (Colab only)')
        args, _ = parser.parse_known_args()
    if hasattr(args, 'gdrive_path') and args.gdrive_path:
        if IN_COLAB:
            print("Mounting Google Drive...")
            gdrive.mount('/content/drive')
            gdrive_path = args.gdrive_path
            if not os.path.exists(gdrive_path):
                print(f"Google Drive file not found: {gdrive_path}")
                sys.exit(1)
            return gdrive_path
        else:
            print("Google Drive path provided, but not running in Colab. Please copy the file locally.")
            sys.exit(1)
    elif hasattr(args, 'local_path') and args.local_path:
        if not os.path.exists(args.local_path):
            print(f"Local file not found: {args.local_path}")
            sys.exit(1)
        return args.local_path
    elif os.path.exists(default_path):
        return default_path
    else:
        print(f"No data file found. Please upload or place a CSV at: {default_path}\nOr use --local-path or --gdrive-path.")
        sys.exit(1)

def robust_read_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(path, encoding='ISO-8859-1', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(path, encoding='ISO-8859-1', on_bad_lines='skip', engine='python')
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        except Exception:
            print('Failed to load data even with all fallbacks. Please check the CSV file.')
            sys.exit(1)
    except Exception:
        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        except Exception:
            print('Failed to load data even with all fallbacks. Please check the CSV file.')
            sys.exit(1)
    if df.empty or len(df.columns) < 2:
        print('Warning: Loaded DataFrame is empty or has too few columns. Data may be corrupted.')
    return df
