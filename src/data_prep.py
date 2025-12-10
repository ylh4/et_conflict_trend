"""
Data preparation and geographic integration module.

This module provides functions to:
- Load Ethiopia administrative boundary shapefiles
- Clean and validate ACLED conflict data
- Perform spatial joins between ACLED events and administrative boundaries
- Handle name normalization for mismatches between ACLED and shapefile names
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.errors import ShapelyError

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.config import ADMIN_BOUNDARIES_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR
from src.utils_logging import get_logger

logger = get_logger(__name__)


# Name normalization mapping for ACLED to shapefile name mismatches
ADMIN1_NAME_MAPPING: Dict[str, str] = {
    "benshangul/gumuz": "Benishangul Gumz",
    "benshangul-gumuz": "Benishangul Gumz",
    "benishangul-gumuz": "Benishangul Gumz",
    "south ethiopia region": "SNNP",
    "south west": "South West Ethiopia",
    "southwest": "South West Ethiopia",
    "central ethiopia": "SNNP",  # May need verification
}


def normalize_admin_name(name: str, admin_level: int = 1) -> str:
    """
    Normalize administrative unit names to match shapefile naming.
    
    Args:
        name: Administrative unit name from ACLED
        admin_level: Administrative level (1, 2, or 3)
    
    Returns:
        Normalized name that should match shapefile names.
    """
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).strip()
    
    # Apply mapping for Admin 1
    if admin_level == 1:
        name_lower = name.lower()
        if name_lower in ADMIN1_NAME_MAPPING:
            return ADMIN1_NAME_MAPPING[name_lower]
    
    # Basic normalization: title case, handle common variations
    name = name.title()
    
    return name


def load_admin0_boundaries() -> gpd.GeoDataFrame:
    """
    Load Admin 0 (country) boundaries for Ethiopia.
    
    Returns:
        GeoDataFrame with Ethiopia country boundary.
    """
    filepath = ADMIN_BOUNDARIES_DIR / "eth_admin0.shp"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Admin 0 shapefile not found at {filepath}. "
            "Please ensure geographic data is in geo/ethiopia_admin_boundaries/"
        )
    
    logger.info(f"Loading Admin 0 boundaries from {filepath}")
    gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf)} country boundary(ies)")
    
    return gdf


def load_admin1_boundaries() -> gpd.GeoDataFrame:
    """
    Load Admin 1 (regional) boundaries for Ethiopia.
    
    Returns:
        GeoDataFrame with 13 regions (Admin 1 units).
    
    Example:
        >>> admin1 = load_admin1_boundaries()
        >>> print(admin1['adm1_name'].tolist())
        ['Addis Ababa', 'Afar', 'Amhara', ...]
    """
    filepath = ADMIN_BOUNDARIES_DIR / "eth_admin1.shp"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Admin 1 shapefile not found at {filepath}. "
            "Please ensure geographic data is in geo/ethiopia_admin_boundaries/"
        )
    
    logger.info(f"Loading Admin 1 boundaries from {filepath}")
    gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf)} regions")
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    
    return gdf


def load_admin2_boundaries() -> gpd.GeoDataFrame:
    """
    Load Admin 2 (zone) boundaries for Ethiopia.
    
    Returns:
        GeoDataFrame with zones (Admin 2 units).
        Each zone includes its parent region in 'adm1_name'.
    
    Example:
        >>> admin2 = load_admin2_boundaries()
        >>> oromia_zones = admin2[admin2['adm1_name'] == 'Oromia']
        >>> print(f"Zones in Oromia: {len(oromia_zones)}")
    """
    filepath = ADMIN_BOUNDARIES_DIR / "eth_admin2.shp"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Admin 2 shapefile not found at {filepath}. "
            "Please ensure geographic data is in geo/ethiopia_admin_boundaries/"
        )
    
    logger.info(f"Loading Admin 2 boundaries from {filepath}")
    gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf)} zones")
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    
    return gdf


def load_admin3_boundaries() -> gpd.GeoDataFrame:
    """
    Load Admin 3 (woreda) boundaries for Ethiopia.
    
    Returns:
        GeoDataFrame with woredas (Admin 3 units).
        Each woreda includes parent zone and region in 'adm2_name' and 'adm1_name'.
    
    Note:
        Admin 3 has 1,082 woredas - use sparingly due to size and complexity.
    """
    filepath = ADMIN_BOUNDARIES_DIR / "eth_admin3.shp"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Admin 3 shapefile not found at {filepath}. "
            "Please ensure geographic data is in geo/ethiopia_admin_boundaries/"
        )
    
    logger.info(f"Loading Admin 3 boundaries from {filepath}")
    gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf)} woredas")
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    
    return gdf


def create_acled_geodataframe(acled_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert ACLED DataFrame to GeoDataFrame using latitude/longitude coordinates.
    
    Args:
        acled_df: DataFrame with ACLED conflict events (must have 'latitude' and 'longitude' columns)
    
    Returns:
        GeoDataFrame with Point geometry for each event.
    
    Raises:
        ValueError: If required columns are missing or coordinates are invalid.
    """
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in acled_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with missing coordinates
    valid_coords = acled_df[required_cols].notna().all(axis=1)
    n_valid = valid_coords.sum()
    n_total = len(acled_df)
    
    if n_valid == 0:
        raise ValueError("No valid coordinates found in ACLED data")
    
    if n_valid < n_total:
        logger.warning(f"Dropping {n_total - n_valid} rows with missing coordinates")
    
    acled_valid = acled_df[valid_coords].copy()
    
    # Create Point geometry
    try:
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(acled_valid['longitude'], acled_valid['latitude'])
        ]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error creating geometry from coordinates: {e}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(acled_valid, geometry=geometry, crs="EPSG:4326")
    
    logger.info(f"Created GeoDataFrame with {len(gdf)} events with valid coordinates")
    
    return gdf


def spatial_join_acled_to_admin(
    acled_df: pd.DataFrame,
    admin_level: int = 1,
    use_name_join: bool = False
) -> pd.DataFrame:
    """
    Join ACLED events to administrative boundaries using spatial or name-based join.
    
    Args:
        acled_df: DataFrame with ACLED conflict events
        admin_level: Administrative level (1, 2, or 3)
        use_name_join: If True, use name-based join instead of spatial join.
                      Spatial join is recommended for accuracy.
    
    Returns:
        DataFrame with ACLED events joined to administrative boundaries.
        Added columns: adm{level}_name, adm{level}_pcode, and other admin attributes.
    """
    # Load admin boundaries
    if admin_level == 1:
        admin_gdf = load_admin1_boundaries()
        admin_name_col = 'adm1_name'
        acled_admin_col = 'admin1'
    elif admin_level == 2:
        admin_gdf = load_admin2_boundaries()
        admin_name_col = 'adm2_name'
        acled_admin_col = 'admin2'
    elif admin_level == 3:
        admin_gdf = load_admin3_boundaries()
        admin_name_col = 'adm3_name'
        acled_admin_col = 'admin3'
    else:
        raise ValueError(f"Invalid admin_level: {admin_level}. Must be 1, 2, or 3.")
    
    if use_name_join:
        # Name-based join (less accurate, but faster)
        logger.info(f"Performing name-based join to Admin {admin_level}")
        
        # Normalize names
        acled_df = acled_df.copy()
        acled_df['_normalized_admin'] = acled_df[acled_admin_col].apply(
            lambda x: normalize_admin_name(x, admin_level) if pd.notna(x) else ""
        )
        
        admin_gdf = admin_gdf.copy()
        admin_gdf['_normalized_admin'] = admin_gdf[admin_name_col].str.strip()
        
        # Join
        result = acled_df.merge(
            admin_gdf[[admin_name_col, 'geometry', 'adm1_name'] + 
                     ([f'adm{admin_level}_pcode'] if f'adm{admin_level}_pcode' in admin_gdf.columns else [])],
            left_on='_normalized_admin',
            right_on='_normalized_admin',
            how='left',
            suffixes=('', '_admin')
        )
        
        # Drop temporary column
        result = result.drop(columns=['_normalized_admin'], errors='ignore')
        
        matched_count = int(result[admin_name_col].notna().sum())
        logger.info(f"Matched {matched_count} of {len(result)} events using name-based join")
        
    else:
        # Spatial join (recommended - more accurate)
        logger.info(f"Performing spatial join to Admin {admin_level}")
        
        # Create GeoDataFrame from ACLED data
        acled_gdf = create_acled_geodataframe(acled_df)
        
        # Perform spatial join
        result_gdf = gpd.sjoin(
            acled_gdf,
            admin_gdf[[admin_name_col, 'geometry', 'adm1_name'] + 
                     ([f'adm{admin_level}_pcode'] if f'adm{admin_level}_pcode' in admin_gdf.columns else [])],
            how='left',
            predicate='within'
        )
        
        # Convert back to DataFrame (drop geometry and index_right if present)
        cols_to_drop = ['geometry', 'index_right']
        result_df = result_gdf.drop(columns=[c for c in cols_to_drop if c in result_gdf.columns], errors='ignore')
        result = pd.DataFrame(result_df)
        
        # Handle potential duplicate columns from join (keep first occurrence)
        if admin_name_col in result.columns:
            if isinstance(result[admin_name_col], pd.DataFrame):
                result[admin_name_col] = result[admin_name_col].iloc[:, 0]
        
        # Count matches
        if admin_name_col in result.columns:
            matched_series = result[admin_name_col].notna()
            if isinstance(matched_series, pd.Series):
                matched_count = matched_series.sum()
                matched_count = int(matched_count) if pd.notna(matched_count) else 0
            else:
                matched_count = 0
        else:
            matched_count = 0
        
        logger.info(f"Matched {matched_count} of {len(result)} events using spatial join")
    
    return result


def load_raw_acled_data(year: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw ACLED data from CSV files.
    
    Args:
        year: If specified, load data for a specific year. Otherwise, load combined file.
    
    Returns:
        DataFrame with raw ACLED data.
    """
    if year:
        filepath = RAW_DATA_DIR / f"acled_ethiopia_{year}.csv"
    else:
        filepath = RAW_DATA_DIR / "acled_ethiopia_all_years.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"ACLED data file not found at {filepath}. "
            "Please run the data download script first."
        )
    
    logger.info(f"Loading ACLED data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} events")
    
    return df


def clean_acled_data(acled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate ACLED conflict data.
    
    Args:
        acled_df: Raw ACLED DataFrame
    
    Returns:
        Cleaned DataFrame with validated data types and filtered invalid records.
    """
    logger.info("Cleaning ACLED data...")
    
    df = acled_df.copy()
    
    # Convert event_date to datetime
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        invalid_dates = df['event_date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropping {invalid_dates} rows with invalid dates")
            df = df[df['event_date'].notna()]
    
    # Ensure year is integer
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    
    # Ensure fatalities is numeric
    if 'fatalities' in df.columns:
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
    
    # Validate coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Check for valid coordinate ranges (Ethiopia bounds approximately)
        valid_lat = (df['latitude'] >= 3.0) & (df['latitude'] <= 15.0)
        valid_lon = (df['longitude'] >= 32.0) & (df['longitude'] <= 48.0)
        
        invalid_coords = (~valid_lat | ~valid_lon).sum()
        if invalid_coords > 0:
            logger.warning(f"Found {invalid_coords} events with coordinates outside Ethiopia bounds")
            # Don't drop, but flag for review
    
    # Remove duplicates if any
    initial_count = len(df)
    df = df.drop_duplicates(subset=['event_id_cnty'], keep='first')
    if len(df) < initial_count:
        logger.info(f"Removed {initial_count - len(df)} duplicate events")
    
    logger.info(f"Cleaned data: {len(df)} events remaining")
    
    return df


def prepare_acled_with_geography(
    year: Optional[int] = None,
    admin_level: int = 1,
    save_interim: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline: Load, clean, and join ACLED data with administrative boundaries.
    
    Args:
        year: If specified, process data for a specific year. Otherwise, process all years.
        admin_level: Administrative level to join (1, 2, or 3)
        save_interim: If True, save cleaned and joined data to interim directory.
    
    Returns:
        DataFrame with cleaned ACLED data joined to administrative boundaries.
    """
    logger.info("=" * 60)
    logger.info("ACLED Data Preparation with Geography")
    logger.info("=" * 60)
    
    # Load raw data
    df = load_raw_acled_data(year=year)
    
    # Clean data
    df_cleaned = clean_acled_data(df)
    
    # Join with administrative boundaries
    df_with_geo = spatial_join_acled_to_admin(df_cleaned, admin_level=admin_level)
    
    # Save interim data if requested
    if save_interim:
        if year:
            output_file = INTERIM_DATA_DIR / f"acled_ethiopia_{year}_cleaned.csv"
        else:
            output_file = INTERIM_DATA_DIR / "acled_ethiopia_all_years_cleaned.csv"
        
        df_with_geo.to_csv(output_file, index=False)
        logger.info(f"Saved cleaned data to {output_file}")
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Total events: {len(df_with_geo)}")
    
    # Count matches properly
    matched_col = df_with_geo[f'adm{admin_level}_name']
    matched_count = matched_col.notna().sum()
    if isinstance(matched_count, pd.Series):
        matched_count = int(matched_count.iloc[0]) if len(matched_count) > 0 else int(matched_count.sum())
    else:
        matched_count = int(matched_count)
    
    logger.info(f"Events with Admin {admin_level} match: {matched_count}")
    logger.info("=" * 60)
    
    return df_with_geo


if __name__ == "__main__":
    """
    Command-line entry point for data preparation pipeline.
    
    Usage:
        python src/data_prep.py [--admin-level 1] [--year YEAR] [--save-interim]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare ACLED data with geographic integration")
    parser.add_argument(
        "--admin-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Administrative level to join (1=regions, 2=zones, 3=woredas)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process data for a specific year (default: all years)"
    )
    parser.add_argument(
        "--save-interim",
        action="store_true",
        help="Save cleaned data to interim directory"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    from src.utils_logging import setup_logging
    logger = setup_logging()
    
    # Run pipeline
    try:
        result = prepare_acled_with_geography(
            year=args.year,
            admin_level=args.admin_level,
            save_interim=args.save_interim
        )
        print(f"\n✓ Successfully processed {len(result)} events")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

