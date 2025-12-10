"""
Visualization module for Ethiopia conflict analysis.

This module provides functions for generating publication-ready figures:
- Monthly time-series plots with cutoff annotations
- Event type composition charts
- Regional distribution maps (choropleths)
- Change maps
- ITS model fit visualizations
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas not available. Map visualizations will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False

from src.config import (
    ABIY_CUTOFF_DATE,
    FIGURES_DIR,
    ADMIN_BOUNDARIES_DIR
)
from src.features import (
    load_processed_data,
    aggregate_by_event_type,
    add_abiy_period_features
)
from src.config import PROCESSED_DATA_DIR
from src.analysis_time_series import (
    prepare_time_series_data,
    calculate_national_monthly_series
)
from src.analysis_spatiotemporal import (
    calculate_regional_pre_post,
    calculate_regional_change,
    merge_regional_data_with_boundaries
)
from src.analysis_statistical import (
    prepare_its_data,
    fit_its_model
)
from src.utils_logging import get_logger

logger = get_logger(__name__)

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_monthly_events(
    df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
    figsize: tuple = (12, 6),
    use_balanced: bool = True
) -> Path:
    """
    Plot monthly conflict events with cutoff annotation.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 1: Monthly conflict events...")
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Prepare monthly time series
    monthly = prepare_time_series_data(df)
    monthly = monthly.sort_values('date')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot monthly events
    ax.plot(monthly['date'], monthly['event_count'], 
            linewidth=2, color='#2E86AB', label='Monthly Events', zorder=3)
    ax.fill_between(monthly['date'], monthly['event_count'], 
                    alpha=0.3, color='#2E86AB', zorder=2)
    
    # Add cutoff line
    cutoff_date = pd.Timestamp(ABIY_CUTOFF_DATE)
    ax.axvline(x=cutoff_date, color='#A23B72', linestyle='--', 
               linewidth=2, label='Abiy Ahmed Inauguration (2018-04-02)', zorder=4)
    
    # Add period shading
    pre_data = monthly[monthly['date'] < cutoff_date]
    post_data = monthly[monthly['date'] >= cutoff_date]
    
    if len(pre_data) > 0:
        ax.axvspan(pre_data['date'].min(), cutoff_date, 
                  alpha=0.1, color='blue', label='Pre-Abiy Period')
    if len(post_data) > 0:
        ax.axvspan(cutoff_date, post_data['date'].max(), 
                  alpha=0.1, color='red', label='Post-Abiy Period')
    
    # Formatting
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Number of Events', fontweight='bold')
    ax.set_title('Monthly Conflict Events in Ethiopia', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / "fig01_monthly_events.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def plot_monthly_fatalities(
    df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
    figsize: tuple = (12, 6),
    use_balanced: bool = True
) -> Path:
    """
    Plot monthly conflict fatalities with cutoff annotation.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 2: Monthly conflict fatalities...")
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Prepare monthly time series
    monthly = prepare_time_series_data(df)
    monthly = monthly.sort_values('date')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot monthly fatalities
    ax.plot(monthly['date'], monthly['fatalities_sum'], 
            linewidth=2, color='#C73E1D', label='Monthly Fatalities', zorder=3)
    ax.fill_between(monthly['date'], monthly['fatalities_sum'], 
                    alpha=0.3, color='#C73E1D', zorder=2)
    
    # Add cutoff line
    cutoff_date = pd.Timestamp(ABIY_CUTOFF_DATE)
    ax.axvline(x=cutoff_date, color='#A23B72', linestyle='--', 
               linewidth=2, label='Abiy Ahmed Inauguration (2018-04-02)', zorder=4)
    
    # Add period shading
    pre_data = monthly[monthly['date'] < cutoff_date]
    post_data = monthly[monthly['date'] >= cutoff_date]
    
    if len(pre_data) > 0:
        ax.axvspan(pre_data['date'].min(), cutoff_date, 
                  alpha=0.1, color='blue', label='Pre-Abiy Period')
    if len(post_data) > 0:
        ax.axvspan(cutoff_date, post_data['date'].max(), 
                  alpha=0.1, color='red', label='Post-Abiy Period')
    
    # Formatting
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Number of Fatalities', fontweight='bold')
    ax.set_title('Monthly Conflict Fatalities in Ethiopia', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / "fig02_monthly_fatalities.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def plot_event_type_composition(
    df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
    figsize: tuple = (12, 8),
    use_balanced: bool = True
) -> Path:
    """
    Plot event type distribution pre vs post Abiy.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 3: Event type composition...")
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Get event type aggregation
    event_type_agg = aggregate_by_event_type(df, groupby_period=True)
    
    # Pivot for plotting
    pivot_df = event_type_agg.pivot(
        index='event_category',
        columns='period',
        values='event_share'
    ).fillna(0)
    
    # Sort by total share
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=True)
    pivot_df = pivot_df.drop('total', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    x = np.arange(len(pivot_df))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, pivot_df['pre_abiy'], width, 
                    label='Pre-Abiy', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(x + width/2, pivot_df['post_abiy'], width, 
                    label='Post-Abiy', color='#C73E1D', alpha=0.8)
    
    # Add value labels
    for i, (pre_val, post_val) in enumerate(zip(pivot_df['pre_abiy'], pivot_df['post_abiy'])):
        if pre_val > 0:
            ax.text(pre_val + 0.5, i - width/2, f'{pre_val:.1f}%', 
                   va='center', fontsize=8)
        if post_val > 0:
            ax.text(post_val + 0.5, i + width/2, f'{post_val:.1f}%', 
                   va='center', fontsize=8)
    
    # Formatting
    ax.set_yticks(x)
    ax.set_yticklabels(pivot_df.index)
    ax.set_xlabel('Share of Total Events (%)', fontweight='bold')
    ax.set_ylabel('Event Category', fontweight='bold')
    ax.set_title('Event Type Distribution: Pre vs Post Abiy Ahmed', 
                fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / "fig03_event_type_composition.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def plot_regional_distribution(
    df: Optional[pd.DataFrame] = None,
    admin_level: int = 1,
    output_file: Optional[Path] = None,
    figsize: tuple = (16, 10),
    use_balanced: bool = True
) -> Path:
    """
    Plot regional distribution maps (choropleths) for pre and post periods.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info(f"Generating Figure 4: Regional distribution (Admin {admin_level})...")
    
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for map visualizations")
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Check if required admin column exists
    admin_col = f'adm{admin_level}_name'
    if admin_col not in df.columns:
        error_msg = (
            f"Admin Level {admin_level} data not found. "
            f"Column '{admin_col}' is missing from the dataset.\n"
            f"To generate Admin Level {admin_level} visualizations, you need to:\n"
            f"1. Run data preparation with Admin Level {admin_level}:\n"
            f"   python src/data_prep.py --admin-level {admin_level} --save-interim\n"
            f"2. Then regenerate processed data with features:\n"
            f"   python src/features.py --admin-level {admin_level} --save-processed"
        )
        logger.warning(error_msg)
        return None  # Return None instead of raising to allow graceful skipping
    
    # Get regional pre/post data
    regional_pre_post = calculate_regional_pre_post(df, admin_level=admin_level)
    
    # Merge with boundaries
    regional_gdf = merge_regional_data_with_boundaries(
        regional_pre_post, admin_level=admin_level
    )
    
    if regional_gdf is None or len(regional_gdf) == 0:
        logger.warning("Could not merge regional data with boundaries. Skipping map.")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Determine font size based on admin level
    if admin_level == 1:
        fontsize = 10
    elif admin_level == 2:
        fontsize = 8
    else:  # admin_level == 3
        fontsize = 6
    
    admin_col = f'adm{admin_level}_name'
    
    # Pre-Abiy map
    pre_data = regional_gdf[regional_gdf['period'] == 'pre_abiy'].copy()
    if len(pre_data) > 0:
        pre_data.plot(column='event_rate_per_month', ax=ax1, 
                     cmap='Blues', legend=True, edgecolor='black', linewidth=0.5)
        ax1.set_title('Pre-Abiy Period\n(Events per Month)', 
                     fontweight='bold', fontsize=12)
        ax1.axis('off')
        
        # Add labels
        for idx, row in pre_data.iterrows():
            if pd.notna(row[admin_col]) and pd.notna(row.geometry):
                try:
                    centroid = row.geometry.centroid
                    ax1.annotate(
                        text=row[admin_col],
                        xy=(centroid.x, centroid.y),
                        ha='center',
                        va='center',
                        fontsize=fontsize,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5)
                    )
                except Exception:
                    pass  # Skip if geometry is invalid
    
    # Post-Abiy map
    post_data = regional_gdf[regional_gdf['period'] == 'post_abiy'].copy()
    if len(post_data) > 0:
        post_data.plot(column='event_rate_per_month', ax=ax2, 
                      cmap='Reds', legend=True, edgecolor='black', linewidth=0.5)
        ax2.set_title('Post-Abiy Period\n(Events per Month)', 
                     fontweight='bold', fontsize=12)
        ax2.axis('off')
        
        # Add labels
        for idx, row in post_data.iterrows():
            if pd.notna(row[admin_col]) and pd.notna(row.geometry):
                try:
                    centroid = row.geometry.centroid
                    ax2.annotate(
                        text=row[admin_col],
                        xy=(centroid.x, centroid.y),
                        ha='center',
                        va='center',
                        fontsize=fontsize,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5)
                    )
                except Exception:
                    pass  # Skip if geometry is invalid
    
    fig.suptitle('Regional Distribution of Conflict Events: Pre vs Post Abiy Ahmed', 
                fontweight='bold', fontsize=14, y=0.98)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / f"fig04_regional_distribution_admin{admin_level}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def plot_regional_change(
    df: Optional[pd.DataFrame] = None,
    admin_level: int = 1,
    output_file: Optional[Path] = None,
    figsize: tuple = (12, 10),
    use_balanced: bool = True
) -> Path:
    """
    Plot change in conflict intensity by region.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        admin_level: Administrative level (1=regions, 2=zones, 3=woredas)
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info(f"Generating Figure 5: Regional change (Admin {admin_level})...")
    
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for map visualizations")
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Check if required admin column exists
    admin_col = f'adm{admin_level}_name'
    if admin_col not in df.columns:
        error_msg = (
            f"Admin Level {admin_level} data not found. "
            f"Column '{admin_col}' is missing from the dataset.\n"
            f"To generate Admin Level {admin_level} visualizations, you need to:\n"
            f"1. Run data preparation with Admin Level {admin_level}:\n"
            f"   python src/data_prep.py --admin-level {admin_level} --save-interim\n"
            f"2. Then regenerate processed data with features:\n"
            f"   python src/features.py --admin-level {admin_level} --save-processed"
        )
        logger.warning(error_msg)
        return None  # Return None instead of raising to allow graceful skipping
    
    # Get regional change data
    regional_change = calculate_regional_change(df, admin_level=admin_level)
    
    # Merge with boundaries using the merge function
    change_gdf = merge_regional_data_with_boundaries(
        regional_change, admin_level=admin_level
    )
    
    if change_gdf is None or len(change_gdf) == 0:
        logger.warning("Could not merge change data with boundaries. Skipping map.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine font size based on admin level
    if admin_level == 1:
        fontsize = 10
    elif admin_level == 2:
        fontsize = 8
    else:  # admin_level == 3
        fontsize = 6
    
    admin_col = f'adm{admin_level}_name'
    
    # Plot change map (diverging colormap)
    # Use change_rate_per_month column from calculate_regional_change
    change_gdf.plot(column='change_rate_per_month', ax=ax, 
                   cmap='RdYlGn_r', legend=True, 
                   edgecolor='black', linewidth=0.5,
                   missing_kwds={'color': 'lightgray'})
    
    # Add labels
    for idx, row in change_gdf.iterrows():
        if pd.notna(row[admin_col]) and pd.notna(row.geometry):
            try:
                centroid = row.geometry.centroid
                ax.annotate(
                    text=row[admin_col],
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    va='center',
                    fontsize=fontsize,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5)
                )
            except Exception:
                pass  # Skip if geometry is invalid
    
    ax.set_title('Change in Conflict Intensity by Region\n(Post - Pre Abiy Period)', 
                fontweight='bold', fontsize=13)
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / f"fig05_regional_change_admin{admin_level}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def plot_its_model_fit(
    df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
    figsize: tuple = (12, 6),
    use_balanced: bool = True
) -> Path:
    """
    Plot ITS model fit overlay on monthly events.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        output_file: Optional output file path. If None, uses default.
        figsize: Figure size (width, height)
        use_balanced: If True, use time-balanced dataset (default: True)
    
    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 6: ITS model fit...")
    
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not available. Skipping ITS model fit plot.")
        return None
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    # Prepare ITS data and fit model
    its_data = prepare_its_data(df)
    its_results = fit_its_model(its_data)
    
    if 'model' not in its_results:
        logger.error("ITS model not found in results. Cannot generate plot.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot observed data
    ax.scatter(its_data['date'], its_data['event_count'], 
              alpha=0.5, color='#2E86AB', s=30, label='Observed', zorder=3)
    
    # Plot fitted values - get from model predictions
    model = its_results['model']
    X = its_data[['time', 'post', 'time_post']].copy()
    X = sm.add_constant(X)  # Add intercept
    fitted = model.predict(X)
    ax.plot(its_data['date'], fitted, 
           linewidth=2, color='#C73E1D', label='Fitted (ITS Model)', zorder=4)
    
    # Add cutoff line
    cutoff_date = pd.Timestamp(ABIY_CUTOFF_DATE)
    ax.axvline(x=cutoff_date, color='#A23B72', linestyle='--', 
               linewidth=2, label='Abiy Ahmed Inauguration (2018-04-02)', zorder=5)
    
    # Add period shading
    pre_data = its_data[its_data['date'] < cutoff_date]
    post_data = its_data[its_data['date'] >= cutoff_date]
    
    if len(pre_data) > 0:
        ax.axvspan(pre_data['date'].min(), cutoff_date, 
                  alpha=0.1, color='blue', label='Pre-Abiy Period')
    if len(post_data) > 0:
        ax.axvspan(cutoff_date, post_data['date'].max(), 
                  alpha=0.1, color='red', label='Post-Abiy Period')
    
    # Formatting
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Number of Events', fontweight='bold')
    ax.set_title('Interrupted Time-Series Model Fit: Monthly Conflict Events', 
                fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Add model statistics text
    r_squared = its_results.get('r_squared', 0)
    model_text = f"R² = {r_squared:.3f}"
    ax.text(0.02, 0.98, model_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = FIGURES_DIR / "fig06_its_model_fit.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved: {output_file}")
    return output_file


def export_all_figures(
    df: Optional[pd.DataFrame] = None,
    admin_level: int = 1,
    use_balanced: bool = True,
    all_admin_levels: bool = False
) -> Dict[str, Path]:
    """
    Export all required figures for the analysis.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        admin_level: Administrative level for regional figures (default: 1)
        use_balanced: If True, use time-balanced dataset (default: True)
        all_admin_levels: If True, generate regional figures for all admin levels (1, 2, 3)
    
    Returns:
        Dictionary mapping figure names to output file paths.
    """
    logger.info("=" * 60)
    logger.info("Exporting All Figures")
    if use_balanced:
        logger.info("Using time-balanced dataset (preserve all post, match pre duration)")
    logger.info("=" * 60)
    
    if df is None:
        df = load_processed_data(use_balanced=use_balanced)
    
    results = {}
    
    # Figure 1: Monthly events
    try:
        results['fig01'] = plot_monthly_events(df, use_balanced=use_balanced)
    except Exception as e:
        logger.error(f"Error generating fig01: {e}")
    
    # Figure 2: Monthly fatalities
    try:
        results['fig02'] = plot_monthly_fatalities(df, use_balanced=use_balanced)
    except Exception as e:
        logger.error(f"Error generating fig02: {e}")
    
    # Figure 3: Event type composition
    try:
        results['fig03'] = plot_event_type_composition(df, use_balanced=use_balanced)
    except Exception as e:
        logger.error(f"Error generating fig03: {e}")
    
    # Figure 4 & 5: Regional distribution and change
    # Generate for specified admin level or all levels
    admin_levels_to_process = [1, 2, 3] if all_admin_levels else [admin_level]
    
    for level in admin_levels_to_process:
        # Figure 4: Regional distribution
        try:
            key = f'fig04_admin{level}'
            result = plot_regional_distribution(
                df, admin_level=level, use_balanced=use_balanced
            )
            if result is not None:
                results[key] = result
            else:
                logger.warning(f"Skipping fig04 (Admin {level}): Data not available")
        except Exception as e:
            logger.error(f"Error generating fig04 (Admin {level}): {e}")
        
        # Figure 5: Regional change
        try:
            key = f'fig05_admin{level}'
            result = plot_regional_change(
                df, admin_level=level, use_balanced=use_balanced
            )
            if result is not None:
                results[key] = result
            else:
                logger.warning(f"Skipping fig05 (Admin {level}): Data not available")
        except Exception as e:
            logger.error(f"Error generating fig05 (Admin {level}): {e}")
    
    # Figure 6: ITS model fit
    try:
        results['fig06'] = plot_its_model_fit(df, use_balanced=use_balanced)
    except Exception as e:
        logger.error(f"Error generating fig06: {e}")
    
    logger.info("=" * 60)
    logger.info("All figures exported successfully!")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    """
    Command-line entry point for visualization.
    """
    import argparse
    from src.utils_logging import setup_logging
    
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument(
        "--admin-level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Administrative level for regional figures (default: 1)"
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=['all', 'fig01', 'fig02', 'fig03', 'fig04', 'fig05', 'fig06'],
        default='all',
        help="Which figure to generate (default: all)"
    )
    parser.add_argument(
        "--use-balanced",
        action="store_true",
        default=True,
        help="Use time-balanced dataset (default: True)"
    )
    parser.add_argument(
        "--no-balanced",
        dest="use_balanced",
        action="store_false",
        help="Use standard processed data instead of balanced"
    )
    parser.add_argument(
        "--all-admin-levels",
        action="store_true",
        help="Generate regional figures for all admin levels (1, 2, 3)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Visualization Module")
    logger.info("=" * 60)
    
    try:
        df = load_processed_data(use_balanced=args.use_balanced)
        
        if args.figure == 'all':
            export_all_figures(
                df, 
                admin_level=args.admin_level, 
                use_balanced=args.use_balanced,
                all_admin_levels=args.all_admin_levels
            )
        else:
            # Generate individual figure
            figure_map = {
                'fig01': lambda d: plot_monthly_events(d, use_balanced=args.use_balanced),
                'fig02': lambda d: plot_monthly_fatalities(d, use_balanced=args.use_balanced),
                'fig03': lambda d: plot_event_type_composition(d, use_balanced=args.use_balanced),
                'fig04': lambda d: plot_regional_distribution(d, admin_level=args.admin_level, use_balanced=args.use_balanced),
                'fig05': lambda d: plot_regional_change(d, admin_level=args.admin_level, use_balanced=args.use_balanced),
                'fig06': lambda d: plot_its_model_fit(d, use_balanced=args.use_balanced)
            }
            
            if args.figure in figure_map:
                figure_map[args.figure](df)
        
        logger.info("=" * 60)
        logger.info("Visualization complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

