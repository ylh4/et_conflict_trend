"""
Export all analysis tables as PNG images.

This script exports all required tables for the blog post as both CSV and PNG.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import TABLES_DIR
from src.features import (
    load_processed_data,
    aggregate_by_event_type,
    aggregate_regional
)
from src.analysis_time_series import calculate_pre_post_statistics
from src.analysis_spatiotemporal import (
    calculate_regional_pre_post,
    calculate_regional_change
)
from src.analysis_statistical import (
    fit_its_model,
    prepare_its_data,
    calculate_pre_post_comparison_table,
    export_its_results,
    export_table_with_png
)
from src.table_export import save_table_as_png_advanced
from src.utils_logging import setup_logging

logger = setup_logging()


def export_table_01_pre_post_overall(df):
    """Export Table 1: Summary of monthly conflict events and fatalities, pre vs post Abiy"""
    logger.info("Exporting Table 1: Pre/post overall summary...")
    
    comparison = calculate_pre_post_comparison_table(df)
    output_file = TABLES_DIR / "tbl01_pre_post_overall.csv"
    
    export_table_with_png(
        comparison,
        output_file,
        title="Table 1: Summary of Monthly Conflict Events and Fatalities",
        subtitle="Pre vs Post Abiy Ahmed Period (2018-04-02)",
        figsize=(14, 6),
        fontsize=11,
    )
    
    return output_file


def export_table_02_event_type_distribution(df):
    """Export Table 2: Event type distribution pre vs post Abiy"""
    logger.info("Exporting Table 2: Event type distribution...")
    
    event_type_agg = aggregate_by_event_type(df, groupby_period=True)
    output_file = TABLES_DIR / "tbl02_event_type_distribution.csv"
    
    # Format for display
    display_df = event_type_agg[['event_category', 'period', 'event_count', 'event_share']].copy()
    display_df['event_share'] = display_df['event_share'].apply(lambda x: f"{x:.1f}%")
    
    export_table_with_png(
        display_df,
        output_file,
        title="Table 2: Event Type Distribution",
        subtitle="Share of total events by type, pre vs post Abiy",
        figsize=(12, 8),
        fontsize=10,
    )
    
    return output_file


def export_table_03_regional_averages(df, admin_level=1):
    """Export Table 3: Regional averages of events per month, pre vs post Abiy"""
    logger.info(f"Exporting Table 3: Regional averages (Admin {admin_level})...")
    
    regional_pre_post = calculate_regional_pre_post(df, admin_level=admin_level)
    output_file = TABLES_DIR / f"tbl03_regional_averages_admin{admin_level}.csv"
    
    # Format for display
    admin_col = f'adm{admin_level}_name'
    display_df = regional_pre_post[[admin_col, 'period', 'event_count', 'event_rate_per_month']].copy()
    display_df['event_rate_per_month'] = display_df['event_rate_per_month'].apply(lambda x: f"{x:.2f}")
    
    export_table_with_png(
        display_df,
        output_file,
        title=f"Table 3: Regional Averages of Events per Month",
        subtitle=f"Pre vs Post Abiy by {admin_col.replace('_', ' ').title()}",
        figsize=(14, 10),
        fontsize=9,
    )
    
    return output_file


def export_table_04_its_coefficients(df):
    """Export Table 4: Interrupted time-series regression coefficients"""
    logger.info("Exporting Table 4: ITS regression coefficients...")
    
    its_data = prepare_its_data(df)
    its_results = fit_its_model(its_data)
    output_file = export_its_results(its_results)
    
    return output_file


def export_all_tables(df=None, admin_level=1):
    """
    Export all required tables as CSV and PNG.
    
    Args:
        df: Optional DataFrame. If None, loads processed data.
        admin_level: Administrative level for regional tables (default: 1)
    
    Returns:
        Dictionary mapping table names to output file paths.
    """
    logger.info("=" * 60)
    logger.info("Exporting All Tables")
    logger.info("=" * 60)
    
    if df is None:
        df = load_processed_data()
    
    results = {}
    
    # Table 1: Pre/post overall
    results['table_01'] = export_table_01_pre_post_overall(df)
    
    # Table 2: Event type distribution
    results['table_02'] = export_table_02_event_type_distribution(df)
    
    # Table 3: Regional averages
    results['table_03'] = export_table_03_regional_averages(df, admin_level=admin_level)
    
    # Table 4: ITS coefficients
    results['table_04'] = export_table_04_its_coefficients(df)
    
    logger.info("=" * 60)
    logger.info("All tables exported successfully!")
    logger.info(f"Tables saved to: {TABLES_DIR}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    """
    Command-line entry point for exporting all tables.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Export all analysis tables as PNG")
    parser.add_argument(
        "--admin-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Administrative level for regional tables"
    )
    
    args = parser.parse_args()
    
    try:
        results = export_all_tables(admin_level=args.admin_level)
        print("\n✓ All tables exported successfully!")
        print(f"  Table 1: {results['table_01']}")
        print(f"  Table 2: {results['table_02']}")
        print(f"  Table 3: {results['table_03']}")
        print(f"  Table 4: {results['table_04']}")
    except Exception as e:
        logger.error(f"Error exporting tables: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

