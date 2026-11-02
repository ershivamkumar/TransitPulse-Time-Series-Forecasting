"""
TransitPulse - Data Download Script
Fetches the official New York MTA Daily Ridership and Traffic dataset.
Saves raw CSV to data/raw/mta_daily_ridership.csv.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_mta_data, load_raw_data, validate_raw_schema


def main():
    print("=" * 60)
    print("TransitPulse: Downloading Official MTA Ridership Dataset")
    print("=" * 60)
    
    raw_path = PROJECT_ROOT / "data" / "raw" / "mta_daily_ridership.csv"
    downloaded_file = download_mta_data(dest_path=raw_path, force_download=True)
    
    df_raw = load_raw_data(downloaded_file)
    is_valid, msg = validate_raw_schema(df_raw)
    
    if not is_valid:
        print(f"[ERROR] Schema validation failed: {msg}")
        sys.exit(1)
        
    print(f"[SUCCESS] Dataset downloaded and validated successfully!")
    print(f"Total raw observations: {len(df_raw):,}")
    print(f"Columns: {list(df_raw.columns)}")
    print(f"Modes represented: {df_raw['mode'].dropna().unique().tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
