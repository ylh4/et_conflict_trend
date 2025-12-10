#!/usr/bin/env python3
"""
Data download script for ACLED Ethiopia conflict data.

This script downloads ACLED conflict events for Ethiopia for the configured
time window and saves raw data to data/raw/.

Usage:
    python scripts/download_acled_data.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import START_YEAR, END_YEAR, validate_credentials
from src.acled_client import fetch_acled_range, load_cached_data
from src.utils_logging import setup_logging

# Set up logging
logger = setup_logging()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ACLED Data Download Script")
    logger.info("=" * 60)
    
    # Validate credentials
    try:
        validate_credentials()
        logger.info("✓ ACLED credentials validated")
    except ValueError as e:
        logger.error(f"✗ Credential validation failed: {e}")
        logger.error("Please create a .env file with ACLED_USERNAME and ACLED_PASSWORD")
        sys.exit(1)
    
    # Check if data already exists
    logger.info(f"\nChecking for existing data...")
    cached_data = load_cached_data()
    
    if not cached_data.empty:
        logger.info(f"Found cached data with {len(cached_data)} records")
        response = input("\nDo you want to re-download data? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Using cached data. Exiting.")
            sys.exit(0)
    
    # Download data
    logger.info(f"\nDownloading ACLED data for Ethiopia")
    logger.info(f"Time window: {START_YEAR} to {END_YEAR}")
    logger.info(f"This may take several minutes depending on data volume...\n")
    
    try:
        df = fetch_acled_range(START_YEAR, END_YEAR, save_individual=True)
        
        if not df.empty:
            logger.info("\n" + "=" * 60)
            logger.info("Download completed successfully!")
            logger.info(f"Total records: {len(df)}")
            logger.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
            logger.info(f"Data saved to: data/raw/")
            logger.info("=" * 60)
        else:
            logger.warning("No data was downloaded. Please check your API credentials and network connection.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        logger.error("Please check your API credentials, network connection, and try again.")
        sys.exit(1)

