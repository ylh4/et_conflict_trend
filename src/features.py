"""
Feature engineering module for Ethiopia conflict analysis.

This module provides functions to:
- Add pre/post Abiy period indicators
- Create monthly and regional aggregations
- Categorize and standardize event types
- Generate analysis-ready features
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import date

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import ABIY_CUTOFF_DATE, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from src.utils_logging import get_logger

logger = get_logger(__name__)


def add_abiy_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre/post Abiy Ahmed period indicators to the dataset.
    
    Classifies each event as occurring before or after Abiy Ahmed was sworn in
    as Prime Minister (2018-04-02).
    
    Args:
        df: DataFrame with ACLED events (must have 'event_date' column)
    
    Returns:
        DataFrame with added columns:
        - 'period': 'pre_abiy' or 'post_abiy'
        - 'is_post_abiy': Boolean (True if post-Abiy, False if pre-Abiy)
        - 'days_since_cutoff': Number of days from cutoff date (negative = before, positive = after)
        - 'years_since_cutoff': Number of years from cutoff date
    
    Example:
        >>> df = add_abiy_period_features(df)
        >>> df['period'].value_counts()
        pre_abiy    5000
        post_abiy   7000
    """
    logger.info("Adding Abiy period features...")
    
    df = df.copy()
    
    # Ensure event_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_date']):
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    # Convert cutoff date to datetime
    cutoff_datetime = pd.Timestamp(ABIY_CUTOFF_DATE)
    
    # Classify periods
    df['is_post_abiy'] = df['event_date'] >= cutoff_datetime
    df['period'] = df['is_post_abiy'].map({True: 'post_abiy', False: 'pre_abiy'})
    
    # Calculate time differences
    df['days_since_cutoff'] = (df['event_date'] - cutoff_datetime).dt.days
    df['years_since_cutoff'] = df['days_since_cutoff'] / 365.25
    
    # Count events by period
    period_counts = df['period'].value_counts()
    logger.info(f"Period classification:")
    logger.info(f"  Pre-Abiy: {period_counts.get('pre_abiy', 0)} events")
    logger.info(f"  Post-Abiy: {period_counts.get('post_abiy', 0)} events")
    
    return df


def categorize_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize and standardize event types.
    
    Groups similar event types and creates standardized categories for analysis.
    
    Args:
        df: DataFrame with ACLED events (must have 'event_type' and/or 'sub_event_type' columns)
    
    Returns:
        DataFrame with added columns:
        - 'event_category': Standardized event category
        - 'is_violent': Boolean indicating if event involves violence
        - 'is_fatal': Boolean indicating if event has fatalities > 0
    """
    logger.info("Categorizing event types...")
    
    df = df.copy()
    
    # Define event type categories
    event_categories = {
        'Battles': ['Violence against civilians', 'Battle-No change of territory', 
                    'Battle-Non-state actor overtakes territory', 'Battle-Government regains territory'],
        'Violence against civilians': ['Violence against civilians'],
        'Protests': ['Protests'],
        'Riots': ['Riots'],
        'Explosions/Remote violence': ['Explosions/Remote violence'],
        'Strategic developments': ['Strategic developments'],
    }
    
    # Create reverse mapping
    type_to_category = {}
    for category, types in event_categories.items():
        for event_type in types:
            type_to_category[event_type] = category
    
    # Categorize based on event_type
    if 'event_type' in df.columns:
        df['event_category'] = df['event_type'].map(type_to_category)
        # Fill missing with original event_type
        df['event_category'] = df['event_category'].fillna(df['event_type'])
    else:
        logger.warning("'event_type' column not found, skipping categorization")
        df['event_category'] = 'Unknown'
    
    # Identify violent events
    violent_categories = ['Battles', 'Violence against civilians', 'Riots', 'Explosions/Remote violence']
    df['is_violent'] = df['event_category'].isin(violent_categories)
    
    # Identify fatal events
    if 'fatalities' in df.columns:
        df['is_fatal'] = df['fatalities'] > 0
    else:
        df['is_fatal'] = False
        logger.warning("'fatalities' column not found, setting is_fatal to False")
    
    # Log category distribution
    if 'event_category' in df.columns:
        category_counts = df['event_category'].value_counts()
        logger.info(f"Event categories: {dict(category_counts)}")
    
    return df


def aggregate_monthly(df: pd.DataFrame, groupby_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Aggregate events to monthly level.
    
    Creates monthly summaries of events and fatalities, optionally grouped by
    administrative units or other categories.
    
    Args:
        df: DataFrame with ACLED events (must have 'event_date' column)
        groupby_cols: Optional list of columns to group by (e.g., ['adm1_name', 'period'])
                     If None, aggregates at national level
    
    Returns:
        DataFrame with monthly aggregates:
        - 'year_month': Year-month string (e.g., '2018-04')
        - 'year': Year
        - 'month': Month
        - 'event_count': Number of events
        - 'fatalities_sum': Total fatalities
        - 'fatalities_mean': Mean fatalities per event
        - Plus any groupby columns
    """
    logger.info("Creating monthly aggregates...")
    
    df = df.copy()
    
    # Ensure event_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_date']):
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    # Extract year and month
    df['year'] = df['event_date'].dt.year
    df['month'] = df['event_date'].dt.month
    df['year_month'] = df['event_date'].dt.to_period('M').astype(str)
    
    # Prepare aggregation columns
    agg_dict = {
        'event_id_cnty': 'count',  # Count events
    }
    
    if 'fatalities' in df.columns:
        agg_dict['fatalities'] = ['sum', 'mean', 'max']
    
    # Group by columns
    group_cols = ['year', 'month', 'year_month']
    if groupby_cols:
        group_cols = groupby_cols + group_cols
    
    # Aggregate
    monthly = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    monthly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in monthly.columns.values]
    
    # Rename columns for clarity
    monthly = monthly.rename(columns={
        'event_id_cnty_count': 'event_count',
        'fatalities_sum': 'fatalities_sum',
        'fatalities_mean': 'fatalities_mean',
        'fatalities_max': 'fatalities_max',
    })
    
    # Sort by date
    monthly = monthly.sort_values(['year', 'month']).reset_index(drop=True)
    
    logger.info(f"Created monthly aggregates: {len(monthly)} month-period combinations")
    
    return monthly


def aggregate_regional(df: pd.DataFrame, admin_level: int = 1, 
                       groupby_period: bool = True) -> pd.DataFrame:
    """
    Aggregate events by administrative region.
    
    Creates regional summaries of events and fatalities, optionally split by
    pre/post Abiy period.
    
    Args:
        df: DataFrame with ACLED events (must have admin boundary columns)
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
        groupby_period: If True, also group by pre/post Abiy period
    
    Returns:
        DataFrame with regional aggregates:
        - 'adm{level}_name': Administrative unit name
        - 'period': Pre/post Abiy (if groupby_period=True)
        - 'event_count': Number of events
        - 'fatalities_sum': Total fatalities
        - 'fatalities_mean': Mean fatalities per event
        - 'event_rate_per_month': Average events per month
    """
    logger.info(f"Creating regional aggregates (Admin {admin_level})...")
    
    df = df.copy()
    
    # Remove duplicate columns (can occur from spatial joins)
    df = df.loc[:, ~df.columns.duplicated()]
    
    admin_col = f'adm{admin_level}_name'
    
    if admin_col not in df.columns:
        raise ValueError(f"Admin {admin_level} column '{admin_col}' not found in DataFrame")
    
    # Prepare groupby columns
    group_cols = [admin_col]
    
    if groupby_period and 'period' in df.columns:
        group_cols.append('period')
    
    # Prepare aggregation
    agg_dict = {
        'event_id_cnty': 'count',
    }
    
    if 'fatalities' in df.columns:
        agg_dict['fatalities'] = ['sum', 'mean', 'max']
    
    # Aggregate
    regional = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    regional.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                       for col in regional.columns.values]
    
    # Rename columns
    regional = regional.rename(columns={
        'event_id_cnty_count': 'event_count',
        'fatalities_sum': 'fatalities_sum',
        'fatalities_mean': 'fatalities_mean',
        'fatalities_max': 'fatalities_max',
    })
    
    # Calculate event rate per month (if period is available)
    if groupby_period and 'period' in regional.columns:
        # Calculate months in each period (approximate)
        # Pre-Abiy: from START_YEAR to cutoff (2013-2018 = ~5 years = 60 months)
        # Post-Abiy: from cutoff to END_YEAR (2018-2025 = ~7 years = 84 months)
        period_months = {
            'pre_abiy': 60,  # Approximate
            'post_abiy': 84,  # Approximate
        }
        regional['period_months'] = regional['period'].map(period_months)
        regional['event_rate_per_month'] = regional['event_count'] / regional['period_months']
    else:
        # Overall rate (approximate total months)
        total_months = 144  # Approximate (2013-2025)
        regional['event_rate_per_month'] = regional['event_count'] / total_months
    
    logger.info(f"Created regional aggregates: {len(regional)} region-period combinations")
    
    return regional


def aggregate_by_event_type(df: pd.DataFrame, groupby_period: bool = True) -> pd.DataFrame:
    """
    Aggregate events by event type/category.
    
    Creates summaries of events by type, optionally split by pre/post Abiy period.
    
    Args:
        df: DataFrame with ACLED events
        groupby_period: If True, also group by pre/post Abiy period
    
    Returns:
        DataFrame with event type aggregates:
        - 'event_category' or 'event_type': Event type/category
        - 'period': Pre/post Abiy (if groupby_period=True)
        - 'event_count': Number of events
        - 'fatalities_sum': Total fatalities
        - 'event_share': Share of total events (%)
    """
    logger.info("Creating event type aggregates...")
    
    df = df.copy()
    
    # Use event_category if available, otherwise event_type
    type_col = 'event_category' if 'event_category' in df.columns else 'event_type'
    
    if type_col not in df.columns:
        raise ValueError(f"Event type column not found. Expected 'event_category' or 'event_type'")
    
    # Prepare groupby columns
    group_cols = [type_col]
    if groupby_period and 'period' in df.columns:
        group_cols.append('period')
    
    # Aggregate
    agg_dict = {
        'event_id_cnty': 'count',
    }
    
    if 'fatalities' in df.columns:
        agg_dict['fatalities'] = ['sum', 'mean']
    
    event_type_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    event_type_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                             for col in event_type_agg.columns.values]
    
    # Rename columns
    event_type_agg = event_type_agg.rename(columns={
        'event_id_cnty_count': 'event_count',
        'fatalities_sum': 'fatalities_sum',
        'fatalities_mean': 'fatalities_mean',
    })
    
    # Calculate event share
    if groupby_period and 'period' in event_type_agg.columns:
        # Calculate share within each period
        for period in event_type_agg['period'].unique():
            period_mask = event_type_agg['period'] == period
            period_total = event_type_agg.loc[period_mask, 'event_count'].sum()
            event_type_agg.loc[period_mask, 'event_share'] = (
                event_type_agg.loc[period_mask, 'event_count'] / period_total * 100
            )
    else:
        # Overall share
        total_events = event_type_agg['event_count'].sum()
        event_type_agg['event_share'] = event_type_agg['event_count'] / total_events * 100
    
    logger.info(f"Created event type aggregates: {len(event_type_agg)} type-period combinations")
    
    return event_type_agg


def create_all_features(df: pd.DataFrame, admin_level: int = 1, 
                       save_processed: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Applies all feature engineering steps in sequence:
    1. Add Abiy period indicators
    2. Categorize event types
    3. Create aggregations (monthly, regional, event type)
    
    Args:
        df: DataFrame with ACLED events (should have admin boundary columns from data_prep)
        admin_level: Administrative level for regional aggregation (1, 2, or 3)
        save_processed: If True, save processed data to data/processed/
    
    Returns:
        DataFrame with all features added
    """
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Add period features
    df = add_abiy_period_features(df)
    
    # Step 2: Categorize event types
    df = categorize_event_types(df)
    
    # Save processed data
    if save_processed:
        output_file = PROCESSED_DATA_DIR / "acled_ethiopia_processed.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    
    logger.info("=" * 60)
    logger.info("Feature engineering complete!")
    logger.info(f"Total events: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info("=" * 60)
    
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load processed ACLED data with all features.
    
    Returns:
        DataFrame with processed ACLED data including all engineered features.
    """
    filepath = PROCESSED_DATA_DIR / "acled_ethiopia_processed.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            "Please run the feature engineering pipeline first."
        )
    
    logger.info(f"Loading processed data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} events with {len(df.columns)} columns")
    
    return df


if __name__ == "__main__":
    """
    Command-line entry point for feature engineering.
    
    Usage:
        python src/features.py [--admin-level 1] [--save-processed]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature engineering for ACLED data")
    parser.add_argument(
        "--admin-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Administrative level for regional aggregation"
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save processed data to data/processed/"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input file path (default: load from interim directory)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    from src.utils_logging import setup_logging
    logger = setup_logging()
    
    # Load data
    if args.input_file:
        logger.info(f"Loading data from {args.input_file}")
        df = pd.read_csv(args.input_file)
    else:
        # Try to load from interim directory
        interim_file = INTERIM_DATA_DIR / "acled_ethiopia_all_years_cleaned.csv"
        if interim_file.exists():
            logger.info(f"Loading data from {interim_file}")
            df = pd.read_csv(interim_file)
        else:
            # Fall back to raw data
            from src.data_prep import load_raw_acled_data, clean_acled_data
            logger.info("Loading raw data and cleaning...")
            df = load_raw_acled_data()
            df = clean_acled_data(df)
    
    # Run feature engineering
    try:
        result = create_all_features(
            df,
            admin_level=args.admin_level,
            save_processed=args.save_processed
        )
        print(f"\n✓ Successfully processed {len(result)} events")
        print(f"  Columns: {list(result.columns)[:10]}...")
    except Exception as e:
        logger.error(f"Error running feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

