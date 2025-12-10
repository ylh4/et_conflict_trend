"""
Spatio-temporal analysis module for Ethiopia conflict data.

This module provides functions for spatial and spatio-temporal analysis:
- Regional event distributions
- Pre/post comparisons by region
- Change maps and spatial visualizations
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import ABIY_CUTOFF_DATE, REPORTS_DIR
from src.features import (
    load_processed_data,
    aggregate_regional,
    add_abiy_period_features
)
from src.data_prep import load_admin1_boundaries
from src.utils_logging import get_logger

logger = get_logger(__name__)


def calculate_regional_distribution(df: pd.DataFrame, admin_level: int = 1) -> pd.DataFrame:
    """
    Calculate regional distribution of events and fatalities.
    
    Args:
        df: DataFrame with ACLED events (must have admin boundary columns)
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
    
    Returns:
        DataFrame with regional aggregates:
        - adm{level}_name: Administrative unit name
        - event_count: Total events
        - fatalities_sum: Total fatalities
        - event_share: Share of total events (%)
    """
    logger.info(f"Calculating regional distribution (Admin {admin_level})...")
    
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Aggregate by region
    regional = aggregate_regional(df, admin_level=admin_level, groupby_period=False)
    
    # Calculate event share
    total_events = regional['event_count'].sum()
    regional['event_share'] = (regional['event_count'] / total_events * 100) if total_events > 0 else 0
    
    logger.info(f"Regional distribution: {len(regional)} regions")
    
    return regional


def calculate_regional_pre_post(df: pd.DataFrame, admin_level: int = 1) -> pd.DataFrame:
    """
    Calculate regional event counts for pre and post Abiy periods.
    
    Args:
        df: DataFrame with ACLED events
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
    
    Returns:
        DataFrame with regional pre/post aggregates:
        - adm{level}_name: Administrative unit name
        - period: 'pre_abiy' or 'post_abiy'
        - event_count: Number of events
        - fatalities_sum: Total fatalities
        - event_rate_per_month: Events per month
    """
    logger.info(f"Calculating regional pre/post comparison (Admin {admin_level})...")
    
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Aggregate by region and period
    regional = aggregate_regional(df, admin_level=admin_level, groupby_period=True)
    
    logger.info(f"Regional pre/post: {len(regional)} region-period combinations")
    
    return regional


def calculate_regional_change(df: pd.DataFrame, admin_level: int = 1) -> pd.DataFrame:
    """
    Calculate change in conflict intensity by region (post - pre).
    
    Args:
        df: DataFrame with ACLED events
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
    
    Returns:
        DataFrame with regional change metrics:
        - adm{level}_name: Administrative unit name
        - pre_events: Events in pre-Abiy period
        - post_events: Events in post-Abiy period
        - change_events: Change in events (post - pre)
        - percent_change_events: Percent change in events
        - pre_rate_per_month: Pre-Abiy events per month
        - post_rate_per_month: Post-Abiy events per month
        - change_rate_per_month: Change in rate per month
    """
    logger.info(f"Calculating regional change metrics (Admin {admin_level})...")
    
    # Get pre/post regional data
    regional_pre_post = calculate_regional_pre_post(df, admin_level=admin_level)
    
    admin_col = f'adm{admin_level}_name'
    
    # Pivot to get pre and post side by side
    pre_data = regional_pre_post[regional_pre_post['period'] == 'pre_abiy'].set_index(admin_col)
    post_data = regional_pre_post[regional_pre_post['period'] == 'post_abiy'].set_index(admin_col)
    
    # Merge pre and post
    change_df = pd.DataFrame(index=regional_pre_post[admin_col].unique())
    change_df['pre_events'] = pre_data['event_count']
    change_df['post_events'] = post_data['event_count']
    change_df['pre_rate_per_month'] = pre_data['event_rate_per_month']
    change_df['post_rate_per_month'] = post_data['event_rate_per_month']
    
    # Fill missing values with 0
    change_df = change_df.fillna(0)
    
    # Calculate changes
    change_df['change_events'] = change_df['post_events'] - change_df['pre_events']
    change_df['change_rate_per_month'] = change_df['post_rate_per_month'] - change_df['pre_rate_per_month']
    
    # Calculate percent change
    change_df['percent_change_events'] = (
        (change_df['post_events'] - change_df['pre_events']) / 
        change_df['pre_events'].replace(0, np.nan) * 100
    ).fillna(0)
    
    # Reset index
    change_df = change_df.reset_index()
    change_df = change_df.rename(columns={'index': admin_col})
    
    logger.info(f"Regional change: {len(change_df)} regions")
    
    return change_df


def merge_regional_data_with_boundaries(
    regional_df: pd.DataFrame,
    admin_level: int = 1
) -> pd.DataFrame:
    """
    Merge regional aggregate data with administrative boundaries for mapping.
    
    Args:
        regional_df: DataFrame with regional aggregates
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
    
    Returns:
        GeoDataFrame with regional data and geometry for mapping.
    """
    logger.info(f"Merging regional data with boundaries (Admin {admin_level})...")
    
    import geopandas as gpd
    
    # Load boundaries
    if admin_level == 1:
        boundaries = load_admin1_boundaries()
        admin_col = 'adm1_name'
    elif admin_level == 2:
        from src.data_prep import load_admin2_boundaries
        boundaries = load_admin2_boundaries()
        admin_col = 'adm2_name'
    elif admin_level == 3:
        from src.data_prep import load_admin3_boundaries
        boundaries = load_admin3_boundaries()
        admin_col = 'adm3_name'
    else:
        raise ValueError(f"Admin level {admin_level} not supported. Use 1, 2, or 3.")
    
    # Merge
    regional_gdf = boundaries[[admin_col, 'geometry']].merge(
        regional_df,
        on=admin_col,
        how='left'
    )
    
    logger.info(f"Merged data: {len(regional_gdf)} regions")
    
    return regional_gdf


def prepare_spatiotemporal_data(
    df: Optional[pd.DataFrame] = None,
    admin_level: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Prepare all spatio-temporal data for analysis and visualization.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
    
    Returns:
        Dictionary with:
        - 'regional_distribution': Overall regional distribution
        - 'regional_pre_post': Regional pre/post comparison
        - 'regional_change': Regional change metrics
    """
    if df is None:
        logger.info("Loading processed data...")
        df = load_processed_data()
    
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    results = {
        'regional_distribution': calculate_regional_distribution(df, admin_level),
        'regional_pre_post': calculate_regional_pre_post(df, admin_level),
        'regional_change': calculate_regional_change(df, admin_level),
    }
    
    return results


if __name__ == "__main__":
    """
    Command-line entry point for spatio-temporal analysis.
    """
    from src.utils_logging import setup_logging
    
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Spatio-Temporal Analysis")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_processed_data()
        
        # Calculate spatio-temporal data
        results = prepare_spatiotemporal_data(df, admin_level=1)
        
        logger.info(f"Regional distribution: {len(results['regional_distribution'])} regions")
        logger.info(f"Regional pre/post: {len(results['regional_pre_post'])} region-periods")
        logger.info(f"Regional change: {len(results['regional_change'])} regions")
        
        logger.info("=" * 60)
        logger.info("Spatio-temporal analysis complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

