"""
TransitPulse - Data Loader Module
Handles downloading, caching, and loading official MTA ridership data.
Source: New York Metropolitan Transportation Authority (MTA)
URL: https://data.ny.gov/d/sayj-mze2
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOCRATA_ENDPOINT = "https://data.ny.gov/resource/sayj-mze2.csv?$limit=50000"
METADATA_ENDPOINT = "https://data.ny.gov/api/views/sayj-mze2/columns.json"

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_RAW_FILE = DEFAULT_RAW_DIR / "mta_daily_ridership.csv"


def fetch_live_metadata() -> list:
    """
    Fetch column metadata from the Socrata API.
    Returns a list of column definitions or empty list on error.
    """
    try:
        response = requests.get(METADATA_ENDPOINT, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning(f"Could not fetch metadata from {METADATA_ENDPOINT}: {exc}")
        return []


def download_mta_data(
    dest_path: Optional[Path] = None,
    force_download: bool = False,
    timeout: int = 30
) -> Path:
    """
    Download the official MTA dataset from NY State Socrata open data portal.
    Caches the CSV locally to avoid repeated network calls.
    """
    dest = Path(dest_path) if dest_path else DEFAULT_RAW_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force_download and dest.stat().st_size > 1024:
        logger.info(f"Using cached raw dataset: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        return dest

    logger.info(f"Downloading MTA ridership dataset from: {SOCRATA_ENDPOINT}")
    try:
        response = requests.get(SOCRATA_ENDPOINT, timeout=timeout)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info(f"Successfully saved {len(response.content)} bytes to {dest}")
        return dest
    except Exception as exc:
        if dest.exists() and dest.stat().st_size > 0:
            logger.warning(f"Network download failed ({exc}), falling back to existing cache: {dest}")
            return dest
        raise RuntimeError(f"Failed to download MTA ridership dataset and no local cache found: {exc}") from exc


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw MTA ridership dataset into a pandas DataFrame.
    """
    path = Path(file_path) if file_path else DEFAULT_RAW_FILE
    if not path.exists():
        logger.info(f"Local file {path} not found. Attempting download...")
        path = download_mta_data(dest_path=path)

    df = pd.read_csv(path)
    logger.info(f"Loaded raw data with shape {df.shape} and columns: {list(df.columns)}")
    return df


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the raw DataFrame contains required columns.
    Expected: date, mode, count
    """
    cols = [c.lower() for c in df.columns]
    required = ["date", "mode", "count"]
    missing = [c for c in required if c not in cols]
    if missing:
        return False, f"Missing required columns in raw dataset: {missing}"
    return True, "Raw schema is valid."


if __name__ == "__main__":
    downloaded_file = download_mta_data()
    df_raw = load_raw_data(downloaded_file)
    valid, msg = validate_raw_schema(df_raw)
    print(f"Validation: {valid} - {msg}")
    print(f"Sample data:\n{df_raw.head()}")
