"""
Time-series analysis module for Ethiopia conflict data.

This module provides functions for temporal trend analysis:
- Monthly national aggregates
- Pre/post period comparisons
- Time-series visualizations
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import ABIY_CUTOFF_DATE, REPORTS_DIR
from src.features import (
    load_processed_data,
    aggregate_monthly,
    add_abiy_period_features
)
from src.utils_logging import get_logger

logger = get_logger(__name__)


def calculate_national_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate national-level monthly time series of events and fatalities.
    
    Args:
        df: DataFrame with ACLED events (must have 'event_date' column)
    
    Returns:
        DataFrame with monthly aggregates:
        - year_month: Year-month string
        - year: Year
        - month: Month
        - event_count: Number of events
        - fatalities_sum: Total fatalities
        - fatalities_mean: Mean fatalities per event
    """
    logger.info("Calculating national monthly time series...")
    
    monthly = aggregate_monthly(df, groupby_cols=None)
    
    logger.info(f"Created monthly series: {len(monthly)} months")
    
    return monthly


def calculate_pre_post_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for pre and post Abiy periods.
    
    Args:
        df: DataFrame with ACLED events (must have 'period' column)
    
    Returns:
        Dictionary with pre/post statistics:
        - pre_events: Total events pre-Abiy
        - post_events: Total events post-Abiy
        - pre_fatalities: Total fatalities pre-Abiy
        - post_fatalities: Total fatalities post-Abiy
        - pre_events_per_month: Average events per month pre-Abiy
        - post_events_per_month: Average events per month post-Abiy
        - pre_fatalities_per_month: Average fatalities per month pre-Abiy
        - post_fatalities_per_month: Average fatalities per month post-Abiy
        - percent_change_events: Percent change in events
        - percent_change_fatalities: Percent change in fatalities
    """
    logger.info("Calculating pre/post Abiy statistics...")
    
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Calculate period durations (approximate)
    # Ensure event_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_date']):
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    pre_start = df[df['period'] == 'pre_abiy']['event_date'].min()
    pre_end = pd.Timestamp(ABIY_CUTOFF_DATE)
    post_start = pd.Timestamp(ABIY_CUTOFF_DATE)
    post_end = df[df['period'] == 'post_abiy']['event_date'].max()
    
    # Convert to Timestamp if needed
    if pd.notna(pre_start):
        pre_start = pd.Timestamp(pre_start)
        pre_days = (pre_end - pre_start).days
    else:
        pre_days = 1825  # ~5 years default
    
    if pd.notna(post_end):
        post_end = pd.Timestamp(post_end)
        post_days = (post_end - post_start).days
    else:
        post_days = 2555  # ~7 years default
    
    pre_months = pre_days / 30.44
    post_months = post_days / 30.44
    
    # Aggregate by period
    period_stats = df.groupby('period').agg({
        'event_id_cnty': 'count',
        'fatalities': 'sum' if 'fatalities' in df.columns else lambda x: 0
    }).to_dict()
    
    pre_events = period_stats['event_id_cnty'].get('pre_abiy', 0)
    post_events = period_stats['event_id_cnty'].get('post_abiy', 0)
    pre_fatalities = period_stats['fatalities'].get('pre_abiy', 0)
    post_fatalities = period_stats['fatalities'].get('post_abiy', 0)
    
    # Calculate rates
    pre_events_per_month = pre_events / pre_months if pre_months > 0 else 0
    post_events_per_month = post_events / post_months if post_months > 0 else 0
    pre_fatalities_per_month = pre_fatalities / pre_months if pre_months > 0 else 0
    post_fatalities_per_month = post_fatalities / post_months if post_months > 0 else 0
    
    # Calculate percent changes
    percent_change_events = ((post_events_per_month - pre_events_per_month) / pre_events_per_month * 100) if pre_events_per_month > 0 else 0
    percent_change_fatalities = ((post_fatalities_per_month - pre_fatalities_per_month) / pre_fatalities_per_month * 100) if pre_fatalities_per_month > 0 else 0
    
    stats = {
        'pre_events': int(pre_events),
        'post_events': int(post_events),
        'pre_fatalities': float(pre_fatalities),
        'post_fatalities': float(post_fatalities),
        'pre_months': float(pre_months),
        'post_months': float(post_months),
        'pre_events_per_month': float(pre_events_per_month),
        'post_events_per_month': float(post_events_per_month),
        'pre_fatalities_per_month': float(pre_fatalities_per_month),
        'post_fatalities_per_month': float(post_fatalities_per_month),
        'percent_change_events': float(percent_change_events),
        'percent_change_fatalities': float(percent_change_fatalities),
    }
    
    logger.info(f"Pre-Abiy: {pre_events} events, {pre_fatalities:.0f} fatalities ({pre_events_per_month:.1f} events/month)")
    logger.info(f"Post-Abiy: {post_events} events, {post_fatalities:.0f} fatalities ({post_events_per_month:.1f} events/month)")
    logger.info(f"Change: {percent_change_events:.1f}% events, {percent_change_fatalities:.1f}% fatalities")
    
    return stats


def prepare_time_series_data(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepare time-series data for analysis and visualization.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
    
    Returns:
        DataFrame with monthly time series ready for visualization.
    """
    if df is None:
        logger.info("Loading processed data...")
        df = load_processed_data()
    
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Calculate monthly series
    monthly = calculate_national_monthly_series(df)
    
    # Add period indicator to monthly data
    monthly['date'] = pd.to_datetime(monthly['year_month'])
    monthly['is_post_abiy'] = monthly['date'] >= pd.Timestamp(ABIY_CUTOFF_DATE)
    monthly['period'] = monthly['is_post_abiy'].map({True: 'post_abiy', False: 'pre_abiy'})
    
    return monthly


if __name__ == "__main__":
    """
    Command-line entry point for time-series analysis.
    """
    from src.utils_logging import setup_logging
    
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Time-Series Analysis")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_processed_data()
        
        # Calculate monthly series
        monthly = prepare_time_series_data(df)
        logger.info(f"Monthly series: {len(monthly)} months")
        
        # Calculate pre/post statistics
        stats = calculate_pre_post_statistics(df)
        
        logger.info("=" * 60)
        logger.info("Time-series analysis complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

