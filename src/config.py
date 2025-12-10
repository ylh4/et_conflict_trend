"""
Central configuration module for Ethiopia conflict pre/post Abiy analysis.

This module centralizes all configuration parameters including paths, dates,
API settings, and analysis parameters.
"""

from datetime import date
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Geographic data directory
GEO_DIR = PROJECT_ROOT / "geo"
ADMIN_BOUNDARIES_DIR = GEO_DIR / "ethiopia_admin_boundaries"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ACLED API configuration
ACLED_BASE_URL = "https://api.acleddata.com/acled/read"
COUNTRY = "Ethiopia"

# Abiy Ahmed cutoff date: sworn in as Prime Minister on 2018-04-02
ABIY_CUTOFF_DATE = date(2018, 4, 2)

# Analysis time window
# Start 5 calendar years before the cutoff date (e.g., 2013 for 2018 cutoff)
START_YEAR = ABIY_CUTOFF_DATE.year - 5
# End year: configurable, default to 2025 (adjust based on ACLED coverage)
END_YEAR = 2025

# ACLED API credentials (loaded from .env)
ACLED_EMAIL: Optional[str] = os.getenv("ACLED_EMAIL")
ACLED_API_KEY: Optional[str] = os.getenv("ACLED_API_KEY")


def validate_credentials() -> None:
    """
    Validate that ACLED credentials are present.
    
    Raises:
        ValueError: If ACLED_EMAIL or ACLED_API_KEY is missing.
    """
    if not ACLED_EMAIL:
        raise ValueError(
            "ACLED_EMAIL not found in environment variables. "
            "Please create a .env file in the project root with ACLED_EMAIL=your_email@example.org"
        )
    if not ACLED_API_KEY:
        raise ValueError(
            "ACLED_API_KEY not found in environment variables. "
            "Please create a .env file in the project root with ACLED_API_KEY=your_api_key"
        )


# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    ADMIN_BOUNDARIES_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

