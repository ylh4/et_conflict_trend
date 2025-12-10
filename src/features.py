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


def balance_pre_post_periods(
    df: pd.DataFrame,
    years_per_period: Optional[int] = None,
    preserve_temporal: bool = True,
    equal_counts: bool = False,
    stratify_by: Optional[list] = None,
    preserve_all_post: bool = False
) -> pd.DataFrame:
    """
    Balance pre and post Abiy periods.
    
    Two modes:
    1. Equal time windows: Selects equal duration periods (e.g., 5 years pre, 5 years post)
    2. Preserve all post: Keeps all post-Abiy data and extends pre-Abiy period backward
       until matching event count (or all available pre data if insufficient)
    
    Preserves temporal patterns within each period.
    
    Args:
        df: DataFrame with ACLED events (must have 'event_date' and 'period' columns)
        years_per_period: Number of years to include in each period (for equal time windows mode)
        preserve_temporal: If True, maintains chronological order (default: True)
        equal_counts: If True, downsample to equal number of events (default: False)
        stratify_by: Optional list of columns to stratify sampling by (e.g., ['adm1_name', 'event_category'])
        preserve_all_post: If True, preserve all post-Abiy data and extend pre backward (default: False)
    
    Returns:
        DataFrame with balanced pre/post periods.
    """
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Ensure event_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_date']):
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    from src.config import ABIY_CUTOFF_DATE
    
    cutoff = pd.Timestamp(ABIY_CUTOFF_DATE)
    
    # Mode 1: Preserve all post, extend pre backward by matching time duration
    if preserve_all_post:
        logger.info("Balancing mode: Preserve all post-Abiy data, match pre-Abiy duration...")
        
        # Get all post-Abiy events
        post_mask = df['event_date'] >= cutoff
        post_events = df[post_mask].copy()
        post_count = len(post_events)
        
        # Calculate post-Abiy time duration
        post_start = cutoff
        post_end = post_events['event_date'].max()
        post_duration = post_end - post_start
        
        logger.info(f"Post-Abiy events (all): {post_count} events")
        logger.info(f"Post-Abiy date range: {post_start.date()} to {post_end.date()}")
        logger.info(f"Post-Abiy duration: {post_duration.days} days ({post_duration.days / 365.25:.2f} years)")
        
        # Set pre-Abiy window to match post duration
        pre_end = cutoff
        pre_start = cutoff - post_duration
        
        # Filter pre-Abiy events within this time window
        pre_mask = (df['event_date'] >= pre_start) & (df['event_date'] < pre_end)
        pre_events = df[pre_mask].copy()
        pre_count = len(pre_events)
        
        logger.info(f"Pre-Abiy window: {pre_start.date()} to {pre_end.date()}")
        logger.info(f"Pre-Abiy duration: {(pre_end - pre_start).days} days ({(pre_end - pre_start).days / 365.25:.2f} years)")
        logger.info(f"Pre-Abiy events: {pre_count} events")
        
    # Mode 2: Equal time windows (original behavior)
    else:
        if years_per_period is None:
            years_per_period = 5
        
        logger.info(f"Balancing mode: Equal time windows ({years_per_period} years per period)...")
        
        # Define time windows
        pre_start = cutoff - pd.DateOffset(years=years_per_period)
        pre_end = cutoff
        post_start = cutoff
        post_end = cutoff + pd.DateOffset(years=years_per_period)
        
        logger.info(f"Pre-Abiy window: {pre_start.date()} to {pre_end.date()}")
        logger.info(f"Post-Abiy window: {post_start.date()} to {post_end.date()}")
        
        # Filter pre-Abiy events
        pre_mask = (df['event_date'] >= pre_start) & (df['event_date'] < pre_end)
        pre_events = df[pre_mask].copy()
        
        # Filter post-Abiy events
        post_mask = (df['event_date'] >= post_start) & (df['event_date'] < post_end)
        post_events = df[post_mask].copy()
    
    # Balance to equal counts if requested
    if equal_counts:
        pre_count = len(pre_events)
        post_count = len(post_events)
        
        if post_count > pre_count:
            # Downsample post-Abiy to match pre-Abiy count
            logger.info(f"Downsampling post-Abiy events from {post_count} to {pre_count}...")
            
            if stratify_by:
                # Stratified sampling
                def sample_stratified(group):
                    n_sample = min(len(group), int(pre_count * len(group) / post_count))
                    return group.sample(n=n_sample, random_state=42) if n_sample > 0 else group.iloc[:0]
                
                post_events = post_events.groupby(stratify_by, group_keys=False, include_groups=False).apply(
                    sample_stratified
                ).reset_index(drop=True)
                # If still too many or too few, adjust to exact count
                if len(post_events) != pre_count:
                    if len(post_events) > pre_count:
                        post_events = post_events.sample(n=pre_count, random_state=42).reset_index(drop=True)
                    else:
                        # If too few, sample with replacement or take all available
                        needed = pre_count - len(post_events)
                        if needed > 0 and len(post_events) > 0:
                            additional = post_events.sample(n=needed, replace=True, random_state=42)
                            post_events = pd.concat([post_events, additional], ignore_index=True)
            else:
                # Simple random sampling
                post_events = post_events.sample(n=pre_count, random_state=42).reset_index(drop=True)
            
            logger.info(f"Post-Abiy events after downsampling: {len(post_events)}")
        elif pre_count > post_count:
            # Downsample pre-Abiy to match post-Abiy count
            logger.info(f"Downsampling pre-Abiy events from {pre_count} to {post_count}...")
            
            if stratify_by:
                # Stratified sampling
                def sample_stratified(group):
                    n_sample = min(len(group), int(post_count * len(group) / pre_count))
                    return group.sample(n=n_sample, random_state=42) if n_sample > 0 else group.iloc[:0]
                
                pre_events = pre_events.groupby(stratify_by, group_keys=False, include_groups=False).apply(
                    sample_stratified
                ).reset_index(drop=True)
                # If still too many or too few, adjust to exact count
                if len(pre_events) != post_count:
                    if len(pre_events) > post_count:
                        pre_events = pre_events.sample(n=post_count, random_state=42).reset_index(drop=True)
                    else:
                        # If too few, sample with replacement or take all available
                        needed = post_count - len(pre_events)
                        if needed > 0 and len(pre_events) > 0:
                            additional = pre_events.sample(n=needed, replace=True, random_state=42)
                            pre_events = pd.concat([pre_events, additional], ignore_index=True)
            else:
                pre_events = pre_events.sample(n=post_count, random_state=42).reset_index(drop=True)
            
            logger.info(f"Pre-Abiy events after downsampling: {len(pre_events)}")
    
    # Combine balanced periods
    balanced_df = pd.concat([pre_events, post_events], ignore_index=True)
    
    # Sort by date if preserving temporal order
    if preserve_temporal:
        balanced_df = balanced_df.sort_values('event_date').reset_index(drop=True)
    
    # Log results
    pre_count = len(pre_events)
    post_count = len(post_events)
    logger.info(f"Balanced dataset:")
    if preserve_all_post:
        # Calculate window durations (not event date ranges)
        post_start_window = cutoff
        post_end_window = post_events['event_date'].max() if len(post_events) > 0 else cutoff
        post_duration = post_end_window - post_start_window
        post_years = post_duration.days / 365.25
        
        pre_end_window = cutoff
        pre_start_window = cutoff - post_duration
        pre_duration = pre_end_window - pre_start_window
        pre_years = pre_duration.days / 365.25
        
        logger.info(f"  Pre-Abiy: {pre_count} events (window: {pre_years:.2f} years)")
        logger.info(f"  Post-Abiy: {post_count} events (window: {post_years:.2f} years)")
        if abs(pre_years - post_years) < 0.01:
            logger.info(f"  ✓ Equal window durations achieved: {pre_years:.2f} years = {post_years:.2f} years")
    else:
        logger.info(f"  Pre-Abiy ({years_per_period} years): {pre_count} events")
        logger.info(f"  Post-Abiy ({years_per_period} years): {post_count} events")
    logger.info(f"  Total: {len(balanced_df)} events")
    if post_count > 0:
        logger.info(f"  Event ratio: {pre_count/post_count:.2f}")
        if equal_counts and pre_count == post_count:
            logger.info(f"  ✓ Equal counts achieved: {pre_count} = {post_count}")
    
    return balanced_df


def create_all_features(df: pd.DataFrame, admin_level: int = 1, 
                       save_processed: bool = True,
                       balance_periods: bool = False,
                       years_per_period: Optional[int] = None,
                       equal_counts: bool = False,
                       stratify_by: Optional[list] = None,
                       preserve_all_post: bool = False) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Applies all feature engineering steps in sequence:
    1. Add Abiy period indicators
    2. Categorize event types
    3. (Optional) Balance pre/post periods for equal time windows
    4. Save processed data
    
    Args:
        df: DataFrame with ACLED events (should have admin boundary columns from data_prep)
        admin_level: Administrative level for regional aggregation (1, 2, or 3)
        save_processed: If True, save processed data to data/processed/
        balance_periods: If True, balance pre/post periods to equal time windows
        years_per_period: Number of years per period when balancing (default: 5)
    
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
    
    # Step 3: Balance periods if requested
    if balance_periods:
        df = balance_pre_post_periods(
            df, 
            years_per_period=years_per_period, 
            preserve_temporal=True,
            equal_counts=equal_counts,
            stratify_by=stratify_by,
            preserve_all_post=preserve_all_post
        )
    
    # Save processed data
    if save_processed:
        if balance_periods:
            if preserve_all_post:
                output_file = PROCESSED_DATA_DIR / "acled_ethiopia_processed_balanced_preserve_post.csv"
            elif equal_counts:
                years_str = f"{years_per_period}yr" if years_per_period else "equal"
                output_file = PROCESSED_DATA_DIR / f"acled_ethiopia_processed_balanced_{years_str}_equal_counts.csv"
            else:
                years_str = f"{years_per_period}yr" if years_per_period else "equal"
                output_file = PROCESSED_DATA_DIR / f"acled_ethiopia_processed_balanced_{years_str}.csv"
        else:
            output_file = PROCESSED_DATA_DIR / "acled_ethiopia_processed.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    
    logger.info("=" * 60)
    logger.info("Feature engineering complete!")
    logger.info(f"Total events: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")
    if balance_periods:
        pre_count = (df['period'] == 'pre_abiy').sum()
        post_count = (df['period'] == 'post_abiy').sum()
        logger.info(f"Pre-Abiy events: {pre_count}")
        logger.info(f"Post-Abiy events: {post_count}")
    logger.info("=" * 60)
    
    return df


def load_processed_data(use_balanced: bool = False) -> pd.DataFrame:
    """
    Load processed ACLED data with all features.
    
    Args:
        use_balanced: If True, load time-balanced dataset (preserve_all_post mode).
                     If False, load standard processed data.
    
    Returns:
        DataFrame with processed ACLED data including all engineered features.
    """
    if use_balanced:
        # Try to load balanced dataset (time-based)
        filepath = PROCESSED_DATA_DIR / "acled_ethiopia_processed_balanced_preserve_post.csv"
        
        if not filepath.exists():
            logger.warning(
                f"Balanced data not found at {filepath}. "
                "Falling back to standard processed data. "
                "To generate balanced data, run: "
                "python src/features.py --balance-periods --preserve-all-post --save-processed"
            )
            filepath = PROCESSED_DATA_DIR / "acled_ethiopia_processed.csv"
    else:
        filepath = PROCESSED_DATA_DIR / "acled_ethiopia_processed.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            "Please run the feature engineering pipeline first."
        )
    
    logger.info(f"Loading processed data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Ensure event_date is datetime
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    logger.info(f"Loaded {len(df)} events with {len(df.columns)} columns")
    
    if use_balanced and "balanced_preserve_post" in str(filepath):
        logger.info("Using time-balanced dataset (preserve all post, match pre duration)")
    
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
        "--balance-periods",
        action="store_true",
        help="Balance pre/post periods to equal time windows"
    )
    parser.add_argument(
        "--years-per-period",
        type=int,
        default=None,
        help="Number of years per period when balancing (for equal time windows mode)"
    )
    parser.add_argument(
        "--equal-counts",
        action="store_true",
        help="Balance to equal number of events (downsample larger period)"
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        nargs='+',
        default=None,
        help="Columns to stratify sampling by (e.g., adm1_name event_category)"
    )
    parser.add_argument(
        "--preserve-all-post",
        action="store_true",
        help="Preserve all post-Abiy data and extend pre-Abiy backward to match count"
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
            save_processed=args.save_processed,
            balance_periods=args.balance_periods,
            years_per_period=args.years_per_period,
            equal_counts=args.equal_counts,
            stratify_by=args.stratify_by,
            preserve_all_post=args.preserve_all_post
        )
        print(f"\n✓ Successfully processed {len(result)} events")
        print(f"  Columns: {list(result.columns)[:10]}...")
    except Exception as e:
        logger.error(f"Error running feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

