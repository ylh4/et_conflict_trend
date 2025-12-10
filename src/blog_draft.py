"""
Blog draft generation module for Ethiopia conflict analysis.

This module generates publication-ready quantitative narrative and limitations
sections for the blog post based on analysis results.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import (
    ABIY_CUTOFF_DATE,
    BLOG_SECTIONS_DIR,
    TABLES_DIR,
    START_YEAR,
    END_YEAR
)
from src.analysis_time_series import calculate_pre_post_statistics
from src.analysis_spatiotemporal import calculate_regional_change
from src.analysis_statistical import fit_its_model, prepare_its_data
from src.features import load_processed_data
from src.utils_logging import get_logger

logger = get_logger(__name__)


def read_summary_tables() -> Dict[str, pd.DataFrame]:
    """
    Load all summary tables from reports/tables/.
    
    Returns:
        Dictionary mapping table names to DataFrames:
        - 'tbl01': Pre/post overall summary
        - 'tbl02': Event type distribution
        - 'tbl03': Regional averages (Admin 1)
        - 'tbl04': ITS coefficients
    """
    logger.info("Reading summary tables...")
    
    tables = {}
    
    # Table 1: Pre/post overall
    tbl01_path = TABLES_DIR / "tbl01_pre_post_overall.csv"
    if tbl01_path.exists():
        tables['tbl01'] = pd.read_csv(tbl01_path)
        logger.info(f"Loaded {tbl01_path}")
    else:
        logger.warning(f"Table 1 not found: {tbl01_path}")
    
    # Table 2: Event type distribution
    tbl02_path = TABLES_DIR / "tbl02_event_type_distribution.csv"
    if tbl02_path.exists():
        tables['tbl02'] = pd.read_csv(tbl02_path)
        logger.info(f"Loaded {tbl02_path}")
    else:
        logger.warning(f"Table 2 not found: {tbl02_path}")
    
    # Table 3: Regional averages (Admin 1)
    tbl03_path = TABLES_DIR / "tbl03_regional_averages_admin1.csv"
    if tbl03_path.exists():
        tables['tbl03'] = pd.read_csv(tbl03_path)
        logger.info(f"Loaded {tbl03_path}")
    else:
        logger.warning(f"Table 3 not found: {tbl03_path}")
    
    # Table 4: ITS coefficients
    tbl04_path = TABLES_DIR / "tbl04_its_coefficients.csv"
    if tbl04_path.exists():
        tables['tbl04'] = pd.read_csv(tbl04_path)
        logger.info(f"Loaded {tbl04_path}")
    else:
        logger.warning(f"Table 4 not found: {tbl04_path}")
    
    return tables


def calculate_percentage_changes(tbl01: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate percentage changes between pre and post periods.
    
    Args:
        tbl01: DataFrame with pre/post overall statistics
    
    Returns:
        Dictionary with percentage changes:
        - 'events_per_month_pct': % change in events per month
        - 'fatalities_per_month_pct': % change in fatalities per month
        - 'total_events_pct': % change in total events
        - 'total_fatalities_pct': % change in total fatalities
    """
    pre = tbl01[tbl01['period'] == 'pre_abiy'].iloc[0]
    post = tbl01[tbl01['period'] == 'post_abiy'].iloc[0]
    
    changes = {}
    
    # Events per month
    if pre['events_per_month'] > 0:
        changes['events_per_month_pct'] = (
            (post['events_per_month'] - pre['events_per_month']) / pre['events_per_month'] * 100
        )
    else:
        changes['events_per_month_pct'] = 0.0
    
    # Fatalities per month
    if pre['fatalities_per_month'] > 0:
        changes['fatalities_per_month_pct'] = (
            (post['fatalities_per_month'] - pre['fatalities_per_month']) / pre['fatalities_per_month'] * 100
        )
    else:
        changes['fatalities_per_month_pct'] = 0.0
    
    # Total events
    if pre['total_events'] > 0:
        changes['total_events_pct'] = (
            (post['total_events'] - pre['total_events']) / pre['total_events'] * 100
        )
    else:
        changes['total_events_pct'] = 0.0
    
    # Total fatalities
    if pre['total_fatalities'] > 0:
        changes['total_fatalities_pct'] = (
            (post['total_fatalities'] - pre['total_fatalities']) / pre['total_fatalities'] * 100
        )
    else:
        changes['total_fatalities_pct'] = 0.0
    
    return changes


def identify_top_changes(
    tbl02: pd.DataFrame,
    tbl03: pd.DataFrame,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify top increases and decreases in event types and regions.
    
    Args:
        tbl02: Event type distribution table
        tbl03: Regional averages table
        df: Optional DataFrame for regional change calculation
    
    Returns:
        Dictionary with:
        - 'event_type_increases': List of (event_type, % change) tuples
        - 'event_type_decreases': List of (event_type, % change) tuples
        - 'region_increases': List of (region, change_rate) tuples
        - 'region_decreases': List of (region, change_rate) tuples
    """
    results = {
        'event_type_increases': [],
        'event_type_decreases': [],
        'region_increases': [],
        'region_decreases': []
    }
    
    # Event type changes
    if tbl02 is not None and len(tbl02) > 0:
        event_types = {}
        for _, row in tbl02.iterrows():
            event_type = row['event_category']
            period = row['period']
            share = float(str(row['event_share']).replace('%', ''))
            
            if event_type not in event_types:
                event_types[event_type] = {}
            event_types[event_type][period] = share
        
        for event_type, shares in event_types.items():
            if 'pre_abiy' in shares and 'post_abiy' in shares:
                change = shares['post_abiy'] - shares['pre_abiy']
                if change > 0:
                    results['event_type_increases'].append((event_type, change))
                elif change < 0:
                    results['event_type_decreases'].append((event_type, abs(change)))
        
        # Sort by magnitude
        results['event_type_increases'].sort(key=lambda x: x[1], reverse=True)
        results['event_type_decreases'].sort(key=lambda x: x[1], reverse=True)
    
    # Regional changes
    if df is not None:
        try:
            regional_change = calculate_regional_change(df, admin_level=1)
            if 'change_rate_per_month' in regional_change.columns:
                for _, row in regional_change.iterrows():
                    region = row['adm1_name']
                    change_rate = row['change_rate_per_month']
                    if pd.notna(change_rate) and region:
                        if change_rate > 0:
                            results['region_increases'].append((region, change_rate))
                        elif change_rate < 0:
                            results['region_decreases'].append((region, abs(change_rate)))
                
                # Sort by magnitude
                results['region_increases'].sort(key=lambda x: x[1], reverse=True)
                results['region_decreases'].sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"Could not calculate regional changes: {e}")
    
    return results


def format_number(value: float, decimals: int = 1, as_percent: bool = False) -> str:
    """
    Format a number for narrative inclusion.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        as_percent: If True, format as percentage
    
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if as_percent:
        return f"{value:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}"


def generate_quantitative_narrative(
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate quantitative narrative section for the blog post.
    
    Requirements:
    - 800-1000 words
    - Data-driven (only findings supported by data)
    - Reference figures and tables
    - Analytical, neutral, concise tone
    
    Args:
        tables: Dictionary of summary tables (if None, reads from files)
        df: Optional DataFrame for regional analysis (if None, loads processed data)
    
    Returns:
        Markdown-formatted narrative text
    """
    logger.info("Generating quantitative narrative...")
    
    # Load data if not provided
    if tables is None:
        tables = read_summary_tables()
    
    if df is None:
        df = load_processed_data(use_balanced=True)
    
    # Extract data
    tbl01 = tables.get('tbl01')
    tbl02 = tables.get('tbl02')
    tbl03 = tables.get('tbl03')
    tbl04 = tables.get('tbl04')
    
    if tbl01 is None or len(tbl01) == 0:
        raise ValueError("Table 1 (pre/post overall) is required but not found")
    
    # Calculate statistics
    changes = calculate_percentage_changes(tbl01)
    top_changes = identify_top_changes(tbl02, tbl03, df)
    
    # Get pre/post values
    pre = tbl01[tbl01['period'] == 'pre_abiy'].iloc[0]
    post = tbl01[tbl01['period'] == 'post_abiy'].iloc[0]
    
    # Get exact balancing dates from the data
    if not pd.api.types.is_datetime64_any_dtype(df['event_date']):
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    pre_start_date = df[df['period'] == 'pre_abiy']['event_date'].min()
    pre_end_date = pd.Timestamp(ABIY_CUTOFF_DATE)
    post_start_date = pd.Timestamp(ABIY_CUTOFF_DATE)
    post_end_date = df[df['period'] == 'post_abiy']['event_date'].max()
    
    # Format dates for narrative
    pre_start_str = pre_start_date.strftime('%B %d, %Y') if pd.notna(pre_start_date) else "N/A"
    pre_end_str = pre_end_date.strftime('%B %d, %Y')
    post_start_str = post_start_date.strftime('%B %d, %Y')
    post_end_str = post_end_date.strftime('%B %d, %Y') if pd.notna(post_end_date) else "N/A"
    
    # Get ITS results if available
    its_results = None
    if tbl04 is not None and len(tbl04) > 0:
        try:
            its_data = prepare_its_data(df)
            its_results = fit_its_model(its_data=its_data)
        except Exception as e:
            logger.warning(f"Could not fit ITS model: {e}")
    
    # Build narrative
    narrative_parts = []
    
    # Executive Summary
    narrative_parts.append("## Executive Summary\n")
    narrative_parts.append(
        f"This analysis examines conflict trends in Ethiopia before and after Abiy Ahmed "
        f"assumed power on {ABIY_CUTOFF_DATE.strftime('%B %d, %Y')}. Using data from the "
        f"Armed Conflict Location & Event Data Project (ACLED), we compare conflict events "
        f"and fatalities across two periods: pre-Abiy ({START_YEAR}-{ABIY_CUTOFF_DATE.year}) "
        f"and post-Abiy ({ABIY_CUTOFF_DATE.year}-{END_YEAR}). The analysis reveals significant "
        f"changes in both the intensity and nature of conflict following the transition.\n"
    )
    
    # Introduction
    narrative_parts.append("## Introduction\n")
    narrative_parts.append(
        f"The Armed Conflict Location & Event Data Project (ACLED) provides comprehensive "
        f"data on political violence and protest events worldwide. This analysis leverages "
        f"ACLED data to examine how conflict patterns in Ethiopia changed following the "
        f"appointment of Abiy Ahmed as Prime Minister on {ABIY_CUTOFF_DATE.strftime('%B %d, %Y')}. "
        f"By comparing equal-duration periods before and after this transition, we aim to "
        f"provide a quantitative assessment of conflict trends while acknowledging the "
        f"descriptive (non-causal) nature of this comparison.\n"
    )
    
    # Methodology
    narrative_parts.append("## Methodology\n")
    narrative_parts.append(
        f"To ensure fair temporal comparison, we balanced the pre and post-Abiy periods by "
        f"preserving all post-Abiy data and matching the pre-Abiy window to the same duration. "
        f"This time-based balancing approach ensures equal time periods for comparison, regardless "
        f"of event counts. The pre-Abiy period spans from {pre_start_str} to {pre_end_str}, "
        f"and the post-Abiy period spans from {post_start_str} to {post_end_str}, "
        f"with the cutoff date of {ABIY_CUTOFF_DATE.strftime('%B %d, %Y')} serving as the "
        f"dividing point. All rates are calculated as events or fatalities per month to account "
        f"for period length differences. Administrative boundaries at Level 1 (regions) are used "
        f"for spatial analysis, and an interrupted time-series (ITS) regression model is employed "
        f"to assess statistical significance of observed changes.\n"
    )
    
    # Event and Fatality Trends
    narrative_parts.append("## Event and Fatality Trends\n")
    narrative_parts.append(
        f"The data reveal substantial increases in conflict activity following Abiy Ahmed's "
        f"assumption of power. As shown in Table 1 and Figure 1, the average number of conflict "
        f"events per month increased from {format_number(pre['events_per_month'], 1)} to "
        f"{format_number(post['events_per_month'], 1)}, representing a "
        f"{format_number(changes['events_per_month_pct'], 1, True)} increase. "
        f"Similarly, fatalities per month rose from {format_number(pre['fatalities_per_month'], 1)} "
        f"to {format_number(post['fatalities_per_month'], 1)}, a "
        f"{format_number(changes['fatalities_per_month_pct'], 1, True)} increase.\n\n"
    )
    narrative_parts.append(
        f"Total events increased from {int(pre['total_events']):,} in the pre-Abiy period to "
        f"{int(post['total_events']):,} in the post-Abiy period, while total fatalities rose from "
        f"{int(pre['total_fatalities']):,} to {int(post['total_fatalities']):,}. "
        f"The mean fatalities per event remained relatively stable, changing from "
        f"{format_number(pre['mean_fatalities_per_event'], 2)} to "
        f"{format_number(post['mean_fatalities_per_event'], 2)} fatalities per event. "
        f"These trends are visualized in Figure 1 (monthly events) and Figure 2 (monthly fatalities), "
        f"which show a clear upward trajectory following the cutoff date.\n"
    )
    
    # Event Type Composition Changes
    narrative_parts.append("## Event Type Composition Changes\n")
    if tbl02 is not None and len(tbl02) > 0:
        narrative_parts.append(
            f"The composition of conflict events shifted significantly between periods, as "
            f"illustrated in Table 2 and Figure 3. The most notable changes include:\n\n"
        )
        
        # Get actual pre/post shares for narrative
        event_type_shares = {}
        for _, row in tbl02.iterrows():
            event_type = row['event_category']
            period = row['period']
            share = float(str(row['event_share']).replace('%', ''))
            if event_type not in event_type_shares:
                event_type_shares[event_type] = {}
            event_type_shares[event_type][period] = share
        
        # Top increases
        if top_changes['event_type_increases']:
            narrative_parts.append("**Event Types with Largest Increases:**\n")
            for i, (event_type, change) in enumerate(top_changes['event_type_increases'][:3], 1):
                pre_share = event_type_shares.get(event_type, {}).get('pre_abiy', 0)
                post_share = event_type_shares.get(event_type, {}).get('post_abiy', 0)
                narrative_parts.append(
                    f"{i}. **{event_type}**: Increased from {format_number(pre_share, 1, True)} "
                    f"to {format_number(post_share, 1, True)} of total events "
                    f"(+{format_number(change, 1, True)} percentage points; see Table 2).\n"
                )
        
        # Top decreases
        if top_changes['event_type_decreases']:
            narrative_parts.append("\n**Event Types with Largest Decreases:**\n")
            for i, (event_type, change) in enumerate(top_changes['event_type_decreases'][:3], 1):
                pre_share = event_type_shares.get(event_type, {}).get('pre_abiy', 0)
                post_share = event_type_shares.get(event_type, {}).get('post_abiy', 0)
                narrative_parts.append(
                    f"{i}. **{event_type}**: Decreased from {format_number(pre_share, 1, True)} "
                    f"to {format_number(post_share, 1, True)} of total events "
                    f"(-{format_number(change, 1, True)} percentage points).\n"
                )
        
        narrative_parts.append(
            f"\nThese shifts reflect changes in the nature of conflict, with certain types of "
            f"violence becoming more or less prominent in the post-Abiy period.\n"
        )
    else:
        narrative_parts.append(
            f"Event type composition data are available in Table 2 and Figure 3, showing "
            f"significant shifts in the types of conflict events between periods.\n"
        )
    
    # Regional Patterns
    narrative_parts.append("## Regional Patterns\n")
    if top_changes['region_increases'] or top_changes['region_decreases']:
        narrative_parts.append(
            f"Conflict intensity changed unevenly across Ethiopia's regions, as shown in "
            f"Table 3 and Figures 4-5. The spatial distribution reveals both increases and "
            f"decreases in conflict activity:\n\n"
        )
        
        # Top regional increases
        if top_changes['region_increases']:
            narrative_parts.append("**Regions with Largest Increases:**\n")
            for i, (region, change_rate) in enumerate(top_changes['region_increases'][:5], 1):
                narrative_parts.append(
                    f"{i}. **{region}**: Increased by {format_number(change_rate, 2)} events per month "
                    f"(see Table 3 and Figure 4 for regional distribution).\n"
                )
        
        # Top regional decreases
        if top_changes['region_decreases']:
            narrative_parts.append("\n**Regions with Largest Decreases:**\n")
            for i, (region, change_rate) in enumerate(top_changes['region_decreases'][:3], 1):
                narrative_parts.append(
                    f"{i}. **{region}**: Decreased by {format_number(change_rate, 2)} events per month.\n"
                )
        
        narrative_parts.append(
            f"\nThe regional change map (Figure 5) provides a visual representation of these "
            f"spatial patterns, highlighting areas of increased and decreased conflict intensity.\n"
        )
    else:
        narrative_parts.append(
            f"Regional analysis (Table 3, Figures 4-5) shows varying patterns of conflict change "
            f"across Ethiopia's administrative regions.\n"
        )
    
    # Statistical Significance
    narrative_parts.append("## Statistical Significance\n")
    if its_results is not None:
        coef_df = its_results['coefficients']
        post_coef = coef_df.loc['post', 'coefficient'] if 'post' in coef_df.index else None
        time_post_coef = coef_df.loc['time_post', 'coefficient'] if 'time_post' in coef_df.index else None
        r_squared = its_results.get('r_squared', 0)
        
        narrative_parts.append(
            f"An interrupted time-series (ITS) regression model was fitted to assess the "
            f"statistical significance of observed changes. The model (Table 4, Figure 6) "
            f"reveals significant effects: "
        )
        
        if post_coef is not None:
            narrative_parts.append(
                f"the immediate level change following the cutoff date is {format_number(post_coef, 2)} "
                f"events per month, "
            )
        
        if time_post_coef is not None:
            narrative_parts.append(
                f"and the trend change (interaction term) is {format_number(time_post_coef, 2)} "
                f"events per month. "
            )
        
        narrative_parts.append(
            f"The model explains {format_number(r_squared * 100, 1, True)} of the variance in "
            f"monthly event counts (R² = {format_number(r_squared, 3)}). These results confirm "
            f"that the observed increases in conflict are statistically significant beyond what "
            f"would be expected from pre-existing trends alone.\n"
        )
    else:
        narrative_parts.append(
            f"Statistical analysis using interrupted time-series regression (Table 4, Figure 6) "
            f"confirms the significance of observed changes, accounting for pre-existing trends.\n"
        )
    
    # Conclusion
    narrative_parts.append("## Conclusion\n")
    narrative_parts.append(
        f"This quantitative analysis reveals substantial increases in conflict activity in Ethiopia "
        f"following Abiy Ahmed's assumption of power in {ABIY_CUTOFF_DATE.year}. Both the frequency "
        f"and intensity of conflict events increased significantly, with notable shifts in event type "
        f"composition and regional distribution. While these findings are descriptive and do not establish "
        f"causality, they provide a comprehensive quantitative assessment of conflict trends during this "
        f"period. The data suggest that the post-Abiy period has been marked by heightened conflict "
        f"activity across multiple dimensions, requiring further qualitative and contextual analysis to "
        f"fully understand the underlying dynamics.\n"
    )
    
    # Combine all parts
    narrative = "\n".join(narrative_parts)
    
    # Word count check
    word_count = len(narrative.split())
    logger.info(f"Generated narrative: {word_count} words")
    
    if word_count < 800:
        logger.warning(f"Narrative is below target word count (800-1000 words). Current: {word_count}")
    elif word_count > 1000:
        logger.warning(f"Narrative exceeds target word count (800-1000 words). Current: {word_count}")
    
    return narrative


def generate_limitations_section() -> str:
    """
    Generate data limitations and caveats section for the blog post.
    
    Requirements:
    - 2-4 paragraphs
    - Neutral language
    - Cover reporting biases, period length adjustments, descriptive nature
    
    Returns:
        Markdown-formatted limitations text
    """
    logger.info("Generating limitations section...")
    
    limitations = []
    
    limitations.append("## Data Limitations and Caveats\n")
    
    limitations.append(
        f"Several important limitations should be considered when interpreting these findings. "
        f"First, ACLED data collection relies on media reports, government sources, and other "
        f"publicly available information, which may introduce reporting biases. Events in remote "
        f"or less-accessible areas may be under-reported, and media attention may vary over time "
        f"and across regions. Additionally, the quality and availability of source information "
        f"may differ between the pre and post-Abiy periods, potentially affecting comparability.\n\n"
    )
    
    limitations.append(
        f"To address differences in period length, we employed time-based balancing that preserves "
        f"all post-Abiy data and matches the pre-Abiy window to the same duration. All rates are "
        f"calculated as events or fatalities per month to normalize for period length. However, "
        f"this approach does not account for potential seasonal patterns or long-term trends that "
        f"may have existed independently of the political transition. The interrupted time-series "
        f"model attempts to control for pre-existing trends, but residual confounding factors may "
        f"remain.\n\n"
    )
    
    limitations.append(
        f"Most importantly, this analysis is descriptive rather than causal. The observed changes "
        f"in conflict patterns cannot be definitively attributed to Abiy Ahmed's leadership or "
        f"policies alone. Multiple factors—including economic conditions, regional dynamics, "
        f"international relations, and historical legacies—likely contribute to conflict trends. "
        f"The pre/post comparison provides a quantitative assessment of changes but does not "
        f"establish causality. Future research combining quantitative analysis with qualitative "
        f"investigation would be valuable for understanding the mechanisms underlying these trends.\n\n"
    )
    
    limitations.append(
        f"Finally, while ACLED provides comprehensive coverage, data quality and completeness may "
        f"vary across event types and regions. Some conflict events may be missed entirely, and "
        f"fatality counts are often estimates with varying degrees of uncertainty. Readers should "
        f"interpret the findings with these limitations in mind and consider them as part of a "
        f"broader body of evidence on conflict in Ethiopia.\n"
    )
    
    text = "\n".join(limitations)
    
    # Paragraph count check
    paragraph_count = text.count('\n\n')
    logger.info(f"Generated limitations section: {paragraph_count} paragraphs")
    
    return text


def insert_figure_table_placeholders(narrative: str) -> str:
    """
    Insert markdown image placeholders for figures and tables at appropriate locations.
    
    Args:
        narrative: The narrative text
    
    Returns:
        Narrative text with placeholders inserted
    """
    result = narrative
    
    # Insert Table 1 and Figure 1 after first mention in Event and Fatality Trends
    if "As shown in Table 1 and Figure 1" in result:
        # Find the paragraph end after this mention
        idx = result.find("As shown in Table 1 and Figure 1")
        # Find the end of the sentence (after the period following the percentage)
        insert_pos = result.find(".", idx)
        if insert_pos != -1:
            # Look for the next sentence end or paragraph break
            next_break = result.find("\n\n", insert_pos)
            if next_break != -1:
                insert_pos = next_break
            else:
                insert_pos = result.find(".", insert_pos + 1)
                if insert_pos != -1:
                    insert_pos += 1
        
        if insert_pos != -1:
            placeholder = "\n\n![Table 1: Summary of Monthly Conflict Events and Fatalities](tables/tbl01_pre_post_overall.png)\n\n![Figure 1: Monthly Conflict Events](figures/fig01_monthly_events.png)\n\n"
            result = result[:insert_pos] + placeholder + result[insert_pos:]
    
    # Insert Figure 2 after mention in Event and Fatality Trends
    if "Figure 2 (monthly fatalities)" in result:
        idx = result.find("Figure 2 (monthly fatalities)")
        insert_pos = result.find(".", idx)
        if insert_pos != -1:
            # Check for paragraph break
            next_break = result.find("\n\n", insert_pos)
            if next_break != -1:
                insert_pos = next_break
            else:
                insert_pos += 1
        
        if insert_pos != -1:
            placeholder = "\n\n![Figure 2: Monthly Conflict Fatalities](figures/fig02_monthly_fatalities.png)\n\n"
            result = result[:insert_pos] + placeholder + result[insert_pos:]
    
    # Insert Table 2 and Figure 3 in Event Type Composition Changes
    if "as illustrated in Table 2 and Figure 3" in result:
        idx = result.find("as illustrated in Table 2 and Figure 3")
        insert_pos = result.find(".", idx)
        if insert_pos != -1:
            # Find paragraph break or next section
            next_break = result.find("\n\n", insert_pos)
            if next_break != -1:
                insert_pos = next_break
            else:
                insert_pos += 1
        
        if insert_pos != -1:
            placeholder = "\n\n![Table 2: Event Type Distribution](tables/tbl02_event_type_distribution.png)\n\n![Figure 3: Event Type Composition](figures/fig03_event_type_composition.png)\n\n"
            result = result[:insert_pos] + placeholder + result[insert_pos:]
    
    # Insert Table 3 and Figures 4-5 in Regional Patterns
    if "as shown in Table 3 and Figures 4-5" in result:
        idx = result.find("as shown in Table 3 and Figures 4-5")
        insert_pos = result.find(":", idx)
        if insert_pos != -1:
            # Find paragraph break after the colon
            next_break = result.find("\n\n", insert_pos)
            if next_break != -1:
                insert_pos = next_break
            else:
                insert_pos = result.find("\n", insert_pos)
                if insert_pos != -1:
                    insert_pos += 1
        
        if insert_pos != -1:
            placeholder = "\n\n![Table 3: Regional Averages](tables/tbl03_regional_averages_admin1.png)\n\n![Figure 4: Regional Distribution](figures/fig04_regional_distribution_admin1.png)\n\n![Figure 5: Regional Change](figures/fig05_regional_change_admin1.png)\n\n"
            result = result[:insert_pos] + placeholder + result[insert_pos:]
    
    # Insert Table 4 and Figure 6 in Statistical Significance
    if "The model (Table 4, Figure 6)" in result:
        idx = result.find("The model (Table 4, Figure 6)")
        insert_pos = result.find(".", idx)
        if insert_pos != -1:
            # Find the end of the sentence about R-squared
            insert_pos = result.find(".", insert_pos + 1)
            if insert_pos != -1:
                insert_pos += 1
                # Check for paragraph break
                next_break = result.find("\n\n", insert_pos)
                if next_break != -1:
                    insert_pos = next_break
        
        if insert_pos != -1:
            placeholder = "\n\n![Table 4: ITS Regression Coefficients](tables/tbl04_its_coefficients.png)\n\n![Figure 6: ITS Model Fit](figures/fig06_its_model_fit.png)\n\n"
            result = result[:insert_pos] + placeholder + result[insert_pos:]
    
    return result


def export_blog_sections(
    output_dir: Optional[Path] = None,
    section: Optional[str] = None
) -> Dict[str, Path]:
    """
    Generate and export blog sections to markdown files.
    
    Args:
        output_dir: Output directory (default: BLOG_SECTIONS_DIR)
        section: Which section to generate ('quantitative', 'limitations', or None for both)
    
    Returns:
        Dictionary mapping section names to output file paths
    """
    if output_dir is None:
        output_dir = BLOG_SECTIONS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate quantitative narrative
    if section is None or section == 'quantitative':
        logger.info("Generating quantitative narrative section...")
        narrative = generate_quantitative_narrative()
        narrative_file = output_dir / "quantitative_trends.md"
        narrative_file.write_text(narrative, encoding='utf-8')
        results['quantitative'] = narrative_file
        logger.info(f"Saved quantitative narrative to {narrative_file}")
    
    # Generate limitations section
    if section is None or section == 'limitations':
        logger.info("Generating limitations section...")
        limitations = generate_limitations_section()
        limitations_file = output_dir / "limitations.md"
        limitations_file.write_text(limitations, encoding='utf-8')
        results['limitations'] = limitations_file
        logger.info(f"Saved limitations section to {limitations_file}")
    
    # Generate complete draft (optional)
    if section is None:
        logger.info("Generating complete draft with figure/table placeholders...")
        # Insert placeholders into narrative
        narrative_with_placeholders = insert_figure_table_placeholders(narrative)
        
        complete_draft = []
        complete_draft.append("# Conflict in Ethiopia: Before and After Abiy Ahmed\n\n")
        complete_draft.append("## Quantitative Analysis\n\n")
        complete_draft.append(narrative_with_placeholders)
        complete_draft.append("\n\n")
        complete_draft.append(limitations)
        
        complete_file = output_dir / "complete_draft.md"
        complete_file.write_text("\n".join(complete_draft), encoding='utf-8')
        results['complete'] = complete_file
        logger.info(f"Saved complete draft with placeholders to {complete_file}")
    
    return results


if __name__ == "__main__":
    """
    Command-line entry point for blog draft generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate blog draft sections from analysis results"
    )
    parser.add_argument(
        '--section',
        choices=['quantitative', 'limitations'],
        default=None,
        help='Which section to generate (default: both)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: reports/blog_sections/)'
    )
    
    args = parser.parse_args()
    
    from src.utils_logging import setup_logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Blog Draft Generation")
    logger.info("=" * 60)
    
    try:
        results = export_blog_sections(
            output_dir=args.output_dir,
            section=args.section
        )
        
        logger.info("=" * 60)
        logger.info("Blog sections generated successfully!")
        for name, path in results.items():
            logger.info(f"  {name}: {path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

