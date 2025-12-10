"""
Table export utilities for publication-ready tables.

This module provides functions to export DataFrames as PNG images
suitable for publication.
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import TABLES_DIR
from src.utils_logging import get_logger

logger = get_logger(__name__)


def save_table_as_png(
    df: pd.DataFrame,
    output_file: Path,
    title: str = "",
    figsize: tuple = (10, 6),
    fontsize: int = 10,
    dpi: int = 300,
    **kwargs
) -> Path:
    """
    Save a DataFrame as a publication-ready PNG table image.
    
    Args:
        df: DataFrame to save as image
        output_file: Output file path (should end with .png)
        title: Optional title for the table
        figsize: Figure size in inches (width, height)
        fontsize: Font size for table text
        dpi: Resolution (dots per inch) for output image
        **kwargs: Additional arguments passed to matplotlib table
    
    Returns:
        Path to saved PNG file.
    """
    logger.info(f"Creating PNG table: {output_file.name}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    # Round numeric columns for better readability
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and abs(x) < 1000 else f"{x:.0f}" if pd.notna(x) else "")
    
    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        rowLabels=df_display.index if df_display.index.name else None,
        cellLoc='center',
        loc='center',
        **kwargs
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(df_display.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style alternating rows
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize + 2, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    
    # Save figure
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info(f"Saved PNG table to {output_file}")
    
    return output_file


def save_table_as_png_advanced(
    df: pd.DataFrame,
    output_file: Path,
    title: str = "",
    subtitle: str = "",
    figsize: tuple = (12, 8),
    fontsize: int = 9,
    dpi: int = 300,
    col_widths: Optional[list] = None,
    highlight_rows: Optional[list] = None,
) -> Path:
    """
    Save a DataFrame as a publication-ready PNG table with advanced styling.
    
    Args:
        df: DataFrame to save as image
        output_file: Output file path (should end with .png)
        title: Main title for the table
        subtitle: Optional subtitle
        figsize: Figure size in inches (width, height)
        fontsize: Font size for table text
        dpi: Resolution (dots per inch) for output image
        col_widths: Optional list of column widths (relative)
        highlight_rows: Optional list of row indices to highlight
    
    Returns:
        Path to saved PNG file.
    """
    logger.info(f"Creating advanced PNG table: {output_file.name}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) and abs(x) < 1000 and abs(x) > 0.001 
            else f"{x:.0f}" if pd.notna(x) and abs(x) >= 1000 
            else f"{x:.4f}" if pd.notna(x) and abs(x) <= 0.001
            else ""
        )
    
    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        rowLabels=df_display.index if df_display.index.name else None,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(len(df_display.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E75B6')
        cell.set_text_props(weight='bold', color='white', size=fontsize + 1)
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # Style data rows
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            cell = table[(i, j)]
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(0.5)
            
            # Alternating row colors
            if i % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('white')
            
            # Highlight specific rows if requested
            if highlight_rows and (i - 1) in highlight_rows:
                cell.set_facecolor('#FFF2CC')
                cell.set_text_props(weight='bold')
    
    # Style index column if present
    if df_display.index.name:
        for i in range(len(df_display) + 1):
            cell = table[(i, -1)]
            if i == 0:
                cell.set_facecolor('#2E75B6')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#E7E6E6')
                cell.set_text_props(weight='bold')
    
    # Position the table in the center, leaving space for title and notes
    # Adjust axes position to center table vertically
    if subtitle:
        # Reserve bottom space for notes
        ax.set_position([0.1, 0.15, 0.8, 0.75])  # [left, bottom, width, height]
    else:
        ax.set_position([0.1, 0.1, 0.8, 0.85])
    
    # Add title (close to table, minimal gap)
    if title:
        fig.suptitle(title, fontsize=fontsize + 4, fontweight='bold', y=0.96)
    
    # Add subtitle/notes at the bottom
    if subtitle:
        # Split subtitle into lines if it contains newlines
        subtitle_lines = subtitle.split('\n')
        num_lines = len(subtitle_lines)
        # Calculate starting position based on number of lines
        y_start = 0.08 - (num_lines - 1) * 0.02
        line_height = 0.02
        
        for i, line in enumerate(subtitle_lines):
            y_pos = y_start - (i * line_height)
            fig.text(0.5, y_pos, line, 
                    ha='center', fontsize=fontsize, style='italic',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', alpha=0.9, 
                             edgecolor='#D0D0D0', linewidth=0.8))
    
    # Final layout adjustment
    plt.tight_layout(rect=[0, 0.10 if subtitle else 0.05, 1, 0.97])
    
    # Save figure
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    logger.info(f"Saved advanced PNG table to {output_file}")
    
    return output_file


if __name__ == "__main__":
    """
    Test table export functionality.
    """
    import pandas as pd
    
    # Create sample table
    test_df = pd.DataFrame({
        'Variable': ['Intercept', 'Time', 'Post', 'Time × Post'],
        'Coefficient': [45.2, 0.15, 12.3, 0.08],
        'Std Error': [2.1, 0.02, 3.4, 0.03],
        'p-value': [0.001, 0.05, 0.01, 0.02]
    })
    
    output_file = TABLES_DIR / "test_table.png"
    save_table_as_png_advanced(
        test_df,
        output_file,
        title="Test Table",
        subtitle="Sample regression results"
    )
    print(f"Test table saved to {output_file}")

