# Ethiopia Administrative Boundaries - Data Structure Guide

## Overview

This directory contains Shapefile data for Ethiopia's administrative boundaries up to the 3rd administrative level (Admin 3 = Woredas).

## Files Available

### Administrative Levels

1. **eth_admin0.shp** - Country level (Ethiopia)
   - 1 polygon
   - Represents the entire country boundary

2. **eth_admin1.shp** - Admin 1 (Regions)
   - **13 regions** (e.g., Addis Ababa, Afar, Amhara, Oromia, Tigray, Somali, etc.)
   - Key columns: `adm1_name`, `adm1_pcode` (e.g., "ET01", "ET05")
   - Area in square kilometers: `area_sqkm`
   - Center coordinates: `center_lat`, `center_lon`

3. **eth_admin2.shp** - Admin 2 (Zones)
   - **92 zones** (e.g., Afder, Agnewak, Alle, Amaro, Arsi, etc.)
   - Key columns: `adm2_name`, `adm2_pcode` (e.g., "ET0101", "ET0508")
   - **Hierarchical:** Contains `adm1_name` (parent region)
   - Area in square kilometers: `area_sqkm`
   - Center coordinates: `center_lat`, `center_lon`

4. **eth_admin3.shp** - Admin 3 (Woredas)
   - **1,082 woredas** (e.g., Tahtay Adiyabo, Laelay Adiabo, Zana, etc.)
   - Key columns: `adm3_name`, `adm3_pcode` (e.g., "ET010101")
   - **Hierarchical:** Contains `adm1_name` (region) and `adm2_name` (zone)
   - Area in square kilometers: `area_sqkm`
   - Center coordinates: `center_lat`, `center_lon`

### Additional Files

5. **eth_admincapitals.shp** - Capital cities
6. **eth_adminlines.shp** - Administrative boundary lines
7. **eth_adminpoints.shp** - Administrative points

## Data Structure

### Common Columns Across All Admin Levels

- `admX_name` - Name of the administrative unit (primary identifier)
- `admX_pcode` - P-code (unique administrative code)
- `adm0_name` - Country name (always "Ethiopia")
- `adm0_pcode` - Country code (always "ET")
- `area_sqkm` - Area in square kilometers
- `center_lat`, `center_lon` - Center coordinates
- `valid_on`, `valid_to` - Validity dates (version v03, valid from 2021-12-14)
- `geometry` - Polygon geometry (GeoPandas)

### Hierarchical Relationships

The shapefiles maintain hierarchical relationships:

- **Admin 2** contains: `adm2_name`, `adm1_name` (parent region)
- **Admin 3** contains: `adm3_name`, `adm2_name` (parent zone), `adm1_name` (parent region)

This means you can:
- Filter zones by region: `admin2[admin2['adm1_name'] == 'Oromia']`
- Filter woredas by region or zone: `admin3[admin3['adm1_name'] == 'Tigray']`

## Coordinate Reference System (CRS)

- **CRS:** EPSG:4326 (WGS84)
- **Format:** Latitude/Longitude (decimal degrees)
- **Ethiopia bounds:**
  - Longitude: 32.99° to 47.99° E
  - Latitude: 3.41° to 14.85° N

## Integration with ACLED Data

### ACLED Data Structure

ACLED conflict data contains:
- `admin1` - Region name (text, e.g., "Oromia", "Tigray")
- `admin2` - Zone name (text, e.g., "West Shewa", "North Western")
- `admin3` - Woreda name (text, e.g., "Ambo town", "Tahtay Adiyabo")
- `latitude`, `longitude` - Event coordinates (WGS84)

### Joining Methods

#### Method 1: Name-Based Join (Recommended for Admin 1 & 2)

```python
import geopandas as gpd
import pandas as pd

# Load shapefiles
admin1 = gpd.read_file("geo/ethiopia_admin_boundaries/eth_admin1.shp")
admin2 = gpd.read_file("geo/ethiopia_admin_boundaries/eth_admin2.shp")

# Load ACLED data
acled = pd.read_csv("data/raw/acled_ethiopia_all_years.csv")

# Join by name (case-insensitive, handle variations)
acled_admin1 = acled.merge(
    admin1[['adm1_name', 'geometry']],
    left_on='admin1',
    right_on='adm1_name',
    how='left'
)
```

**Note:** Name matching may require normalization:
- Handle case differences
- Handle variations (e.g., "Benishangul Gumz" vs "Benishangul-Gumuz")
- Handle special characters

#### Method 2: Spatial Join (Point-in-Polygon) - Recommended for Precision

```python
import geopandas as gpd
from shapely.geometry import Point

# Load shapefiles
admin1 = gpd.read_file("geo/ethiopia_admin_boundaries/eth_admin1.shp")

# Load ACLED data
acled = pd.read_csv("data/raw/acled_ethiopia_all_years.csv")

# Create GeoDataFrame from ACLED coordinates
acled_gdf = gpd.GeoDataFrame(
    acled,
    geometry=[Point(xy) for xy in zip(acled.longitude, acled.latitude)],
    crs='EPSG:4326'
)

# Spatial join (point in polygon)
acled_with_admin1 = gpd.sjoin(
    acled_gdf,
    admin1[['adm1_name', 'geometry']],
    how='left',
    predicate='within'
)
```

**Advantages:**
- More accurate (uses actual coordinates)
- Handles name mismatches
- Works even if ACLED admin names don't match exactly

## Usage Examples

### Example 1: Load Admin 1 Boundaries

```python
import geopandas as gpd
from pathlib import Path

geo_dir = Path("geo/ethiopia_admin_boundaries")
admin1 = gpd.read_file(geo_dir / "eth_admin1.shp")

print(f"Regions: {admin1['adm1_name'].tolist()}")
print(f"Total regions: {len(admin1)}")
```

### Example 2: Filter by Region

```python
# Get all zones in Oromia region
admin2 = gpd.read_file(geo_dir / "eth_admin2.shp")
oromia_zones = admin2[admin2['adm1_name'] == 'Oromia']
print(f"Zones in Oromia: {oromia_zones['adm2_name'].tolist()}")
```

### Example 3: Spatial Join ACLED Events to Admin 1

```python
import geopandas as gpd
from shapely.geometry import Point

# Load data
admin1 = gpd.read_file(geo_dir / "eth_admin1.shp")
acled = pd.read_csv("data/raw/acled_ethiopia_all_years.csv")

# Create point geometry from ACLED coordinates
acled_gdf = gpd.GeoDataFrame(
    acled,
    geometry=[Point(xy) for xy in zip(acled.longitude, acled.latitude)],
    crs='EPSG:4326'
)

# Spatial join
acled_admin1 = gpd.sjoin(acled_gdf, admin1[['adm1_name', 'geometry']], 
                         how='left', predicate='within')

# Count events by region
events_by_region = acled_admin1.groupby('adm1_name').size()
```

## Important Notes

1. **Name Matching:** ACLED admin names may not exactly match shapefile names. Consider:
   - Case normalization
   - Handling special characters
   - Handling variations (e.g., "Benishangul Gumz" vs "Benishangul-Gumuz")

2. **Coordinate Precision:** ACLED coordinates may have varying precision. Spatial joins handle this better than name-based joins.

3. **Missing Data:** Some ACLED events may not have admin1/admin2/admin3 names but have coordinates. Use spatial joins to assign them.

4. **Version:** Shapefiles are version v03, valid from 2021-12-14. Administrative boundaries may have changed over time.

5. **Performance:** Admin 3 (1,082 woredas) spatial joins can be slow. Consider:
   - Filtering to relevant regions first
   - Using spatial indexes
   - Caching results

## Recommended Approach for Analysis

For the pre/post Abiy analysis:

1. **Admin 1 (Regions):** Use for regional-level analysis and choropleth maps
   - 13 regions is manageable for visualization
   - Good for high-level trends

2. **Admin 2 (Zones):** Use for more granular analysis if needed
   - 92 zones may be too detailed for some visualizations
   - Useful for specific regional deep-dives

3. **Admin 3 (Woredas):** Use sparingly due to complexity
   - 1,082 woredas is very granular
   - Consider aggregating to Admin 1 or 2 for most analyses

## Next Steps

When implementing `src/data_prep.py`, create functions:
- `load_admin1_boundaries()` - Returns GeoDataFrame of regions
- `load_admin2_boundaries()` - Returns GeoDataFrame of zones (optional)
- `load_admin3_boundaries()` - Returns GeoDataFrame of woredas (optional)
- `spatial_join_acled_to_admin(acled_df, admin_level=1)` - Spatial join function

