"""
Statistical analysis module for Ethiopia conflict data.

This module provides functions for statistical analysis:
- Interrupted time-series (ITS) regression
- Pre/post period comparisons
- Statistical significance testing
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

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Statistical analysis functions will be limited.")

from src.config import ABIY_CUTOFF_DATE, REPORTS_DIR, TABLES_DIR
from src.features import (
    load_processed_data,
    aggregate_monthly,
    add_abiy_period_features
)
from src.analysis_time_series import prepare_time_series_data
from src.table_export import save_table_as_png_advanced
from src.utils_logging import get_logger

logger = get_logger(__name__)


def prepare_its_data(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepare data for interrupted time-series (ITS) regression.
    
    Creates monthly time series with variables needed for ITS analysis:
    - time: Time trend (months since start)
    - post: Post-Abiy indicator (0=pre, 1=post)
    - time_post: Interaction term (time * post)
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
    
    Returns:
        DataFrame with ITS variables and outcome (event_count).
    """
    logger.info("Preparing data for interrupted time-series regression...")
    
    # Get monthly time series
    monthly = prepare_time_series_data(df)
    
    # Create time variable (months since start)
    monthly = monthly.sort_values('date')
    monthly['time'] = range(len(monthly))
    
    # Create post indicator
    monthly['post'] = (monthly['date'] >= pd.Timestamp(ABIY_CUTOFF_DATE)).astype(int)
    
    # Create interaction term
    monthly['time_post'] = monthly['time'] * monthly['post']
    
    # Ensure we have outcome variable
    if 'event_count' not in monthly.columns:
        raise ValueError("event_count column not found in monthly data")
    
    logger.info(f"ITS data prepared: {len(monthly)} months")
    logger.info(f"  Pre-Abiy months: {(monthly['post'] == 0).sum()}")
    logger.info(f"  Post-Abiy months: {(monthly['post'] == 1).sum()}")
    
    return monthly


def fit_its_model(
    its_data: Optional[pd.DataFrame] = None,
    outcome_var: str = 'event_count'
) -> Dict:
    """
    Fit interrupted time-series (ITS) regression model.
    
    Model: outcome = β₀ + β₁*time + β₂*post + β₃*time_post + ε
    
    Where:
    - β₀: Intercept (baseline level)
    - β₁: Pre-Abiy time trend
    - β₂: Immediate change at cutoff (level change)
    - β₃: Change in trend after cutoff (slope change)
    
    Args:
        its_data: Optional DataFrame with ITS variables. If None, prepares data.
        outcome_var: Outcome variable name (default: 'event_count')
    
    Returns:
        Dictionary with:
        - 'model': Fitted model object
        - 'results': Regression results summary
        - 'coefficients': DataFrame with coefficients and statistics
        - 'r_squared': R-squared value
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ITS regression. Please install: pip install statsmodels")
    
    logger.info("Fitting interrupted time-series regression model...")
    
    if its_data is None:
        its_data = prepare_its_data()
    
    # Prepare data for regression
    X = its_data[['time', 'post', 'time_post']].copy()
    X = sm.add_constant(X)  # Add intercept
    y = its_data[outcome_var].copy()
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Extract coefficients
    coefficients = pd.DataFrame({
        'coefficient': model.params,
        'std_err': model.bse,
        't_stat': model.tvalues,
        'p_value': model.pvalues,
        'conf_int_lower': model.conf_int()[0],
        'conf_int_upper': model.conf_int()[1],
    })
    
    results = {
        'model': model,
        'results': model.summary(),
        'coefficients': coefficients,
        'r_squared': model.rsquared,
        'n_observations': len(its_data),
    }
    
    logger.info(f"ITS model fitted: R² = {model.rsquared:.3f}")
    logger.info(f"  Intercept: {model.params['const']:.2f}")
    logger.info(f"  Time trend (pre): {model.params['time']:.2f}")
    logger.info(f"  Level change (post): {model.params['post']:.2f}")
    logger.info(f"  Trend change (time_post): {model.params['time_post']:.2f}")
    
    return results


def export_table_with_png(
    df: pd.DataFrame,
    output_file: Path,
    title: str = "",
    subtitle: str = "",
    index: bool = False,
    **png_kwargs
) -> Path:
    """
    Export table as both CSV and PNG.
    
    Args:
        df: DataFrame to export
        output_file: Output file path (CSV)
        title: Title for PNG table
        subtitle: Subtitle for PNG table
        index: Whether to include index in CSV export
        **png_kwargs: Additional arguments for PNG export
    
    Returns:
        Path to saved CSV file.
    """
    # Save CSV
    df.to_csv(output_file, index=index)
    logger.info(f"Saved table to {output_file}")
    
    # Save PNG
    png_file = output_file.with_suffix('.png')
    save_table_as_png_advanced(
        df,
        png_file,
        title=title,
        subtitle=subtitle,
        **png_kwargs
    )
    
    return output_file


def calculate_pre_post_comparison_table(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create summary table comparing pre and post Abiy periods.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
    
    Returns:
        DataFrame with pre/post comparison statistics.
    """
    logger.info("Creating pre/post comparison table...")
    
    if df is None:
        df = load_processed_data()
    
    # Ensure period features exist
    if 'period' not in df.columns:
        df = add_abiy_period_features(df)
    
    # Calculate statistics by period
    comparison = df.groupby('period').agg({
        'event_id_cnty': 'count',
        'fatalities': ['sum', 'mean', 'max'] if 'fatalities' in df.columns else 'count',
    }).reset_index()
    
    # Flatten column names
    comparison.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                         for col in comparison.columns.values]
    
    # Rename
    comparison = comparison.rename(columns={
        'event_id_cnty_count': 'total_events',
        'fatalities_sum': 'total_fatalities',
        'fatalities_mean': 'mean_fatalities_per_event',
        'fatalities_max': 'max_fatalities_single_event',
    })
    
    # Calculate rates per month (approximate)
    pre_months = 60  # ~5 years
    post_months = 84  # ~7 years
    
    comparison['events_per_month'] = comparison['total_events'] / [pre_months, post_months]
    comparison['fatalities_per_month'] = comparison['total_fatalities'] / [pre_months, post_months]
    
    logger.info("Pre/post comparison table created")
    
    return comparison


def export_its_results(its_results: Dict, output_file: Optional[Path] = None) -> Path:
    """
    Export ITS regression results to CSV and markdown table.
    
    Args:
        its_results: Dictionary from fit_its_model()
        output_file: Optional output file path. If None, uses default location.
    
    Returns:
        Path to saved file.
    """
    if output_file is None:
        output_file = TABLES_DIR / "tbl04_its_coefficients.csv"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save coefficients as CSV
    its_results['coefficients'].to_csv(output_file, index=True)
    logger.info(f"Saved ITS coefficients to {output_file}")
    
    # Save as PNG table
    png_file = output_file.with_suffix('.png')
    # Format subtitle with model info at bottom
    model_info = f"Model: event_count = β₀ + β₁×time + β₂×post + β₃×time×post + ε"
    stats_info = f"R² = {its_results['r_squared']:.3f} | N = {its_results['n_observations']}"
    subtitle = f"{model_info}\n{stats_info}"
    
    save_table_as_png_advanced(
        its_results['coefficients'],
        png_file,
        title="Interrupted Time-Series Regression Coefficients",
        subtitle=subtitle,
        figsize=(12, 7),
        fontsize=10,
    )
    
    # Also save as markdown (if tabulate is available, otherwise use simple text format)
    md_file = output_file.with_suffix('.md')
    try:
        with open(md_file, 'w') as f:
            f.write("# Interrupted Time-Series Regression Coefficients\n\n")
            f.write("Model: event_count = β₀ + β₁*time + β₂*post + β₃*time_post + ε\n\n")
            # Try to use to_markdown, fallback to simple text if tabulate is missing
            try:
                f.write(its_results['coefficients'].to_markdown())
            except ImportError:
                # Fallback: create simple markdown table manually
                f.write("| Variable | Coefficient | Std Err | t-stat | p-value |\n")
                f.write("|----------|-------------|---------|--------|----------|\n")
                for idx, row in its_results['coefficients'].iterrows():
                    f.write(f"| {idx} | {row['coefficient']:.4f} | {row['std_err']:.4f} | "
                           f"{row['t_stat']:.4f} | {row['p_value']:.4f} |\n")
            f.write(f"\n\nR² = {its_results['r_squared']:.3f}\n")
            f.write(f"N = {its_results['n_observations']}\n")
        
        logger.info(f"Saved ITS results (markdown) to {md_file}")
    except Exception as e:
        logger.warning(f"Could not save markdown file: {e}. CSV file saved successfully.")
    
    return output_file


if __name__ == "__main__":
    """
    Command-line entry point for statistical analysis.
    """
    from src.utils_logging import setup_logging
    
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Statistical Analysis")
    logger.info("=" * 60)
    
    try:
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels not available. Please install: pip install statsmodels")
            sys.exit(1)
        
        # Prepare ITS data
        its_data = prepare_its_data()
        
        # Fit ITS model
        its_results = fit_its_model(its_data)
        
        # Export results
        export_its_results(its_results)
        
        # Create pre/post comparison table
        comparison = calculate_pre_post_comparison_table()
        comparison_file = TABLES_DIR / "tbl01_pre_post_overall.csv"
        export_table_with_png(
            comparison,
            comparison_file,
            title="Summary of Conflict Events and Fatalities: Pre vs Post Abiy",
            subtitle="Monthly averages and total counts",
            figsize=(14, 6),
            fontsize=11,
        )
        
        logger.info("=" * 60)
        logger.info("Statistical analysis complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

