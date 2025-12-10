# Conflict in Ethiopia: Before and After Abiy Ahmed

A comprehensive data analysis pipeline for analyzing conflict trends in Ethiopia before and after Abiy Ahmed's assumption of power (April 2, 2018), using ACLED (Armed Conflict Location & Event Data Project) data.

## Overview

This project implements a reproducible Python analysis stack that:
- Downloads and processes ACLED conflict data for Ethiopia
- Integrates geographic data (administrative boundaries)
- Engineers features with time-based period balancing
- Analyzes temporal trends (time-series analysis)
- Examines spatial patterns (spatio-temporal analysis)
- Performs statistical analysis (interrupted time-series regression)
- Generates publication-quality figures and tables (CSV, Markdown, PNG) for blog publication

## Project Structure

```
et_conflict_trend/
├── data/
│   ├── raw/              # Raw ACLED data (CSV files by year)
│   ├── interim/          # Cleaned data with geographic integration
│   └── processed/        # Feature-engineered data (balanced/unbalanced)
├── geo/
│   └── ethiopia_admin_boundaries/  # Administrative boundary shapefiles (Admin 0-3)
│       ├── eth_admin0.shp         # Country boundary
│       ├── eth_admin1.shp         # Regional boundaries
│       ├── eth_admin2.shp         # Zone boundaries
│       ├── eth_admin3.shp         # Woreda boundaries
│       └── README.md               # Geographic data documentation
├── notebooks/
│   └── 01_download_acled.ipynb   # Interactive data download notebook
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central configuration
│   ├── utils_logging.py          # Logging utilities
│   ├── acled_client.py           # ACLED API client (OAuth authentication)
│   ├── data_prep.py              # Data cleaning and geographic integration
│   ├── features.py               # Feature engineering with period balancing
│   ├── analysis_time_series.py   # Time-series analysis
│   ├── analysis_spatiotemporal.py # Spatial and spatio-temporal analysis
│   ├── analysis_statistical.py   # Statistical analysis (ITS regression)
│   ├── table_export.py            # Publication-ready PNG table export
│   └── export_all_tables.py      # Batch table export script
├── scripts/
│   └── download_acled_data.py    # Standalone data download script
├── reports/
│   ├── figures/                  # Generated figures (PNG)
│   └── tables/                   # Generated tables (CSV, Markdown, PNG)
├── .env                          # ACLED API credentials (create from .env.example)
├── .env.example                  # Template for environment variables
├── requirements.txt             # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A myACLED account at [https://developer.acleddata.com/](https://developer.acleddata.com/)

### 2. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
# Navigate to project root
cd et_conflict_trend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure ACLED API Credentials

ACLED now uses **OAuth 2.0 token-based authentication**. You need your myACLED account credentials:

1. Register for a myACLED account at [https://developer.acleddata.com/](https://developer.acleddata.com/)
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your credentials:
   ```
   ACLED_USERNAME=your_myacled_email@example.org
   ACLED_PASSWORD=your_myacled_password
   ```

**Important:** 
- Never commit the `.env` file to version control. It is already included in `.gitignore`.
- Use your myACLED account email and password (not an API key).

### 5. Prepare Geographic Data

Administrative boundary shapefiles should be placed in:
```
geo/ethiopia_admin_boundaries/
```

The project expects shapefiles for administrative levels 0-3:
- `eth_admin0.shp` - Country boundary
- `eth_admin1.shp` - Regional boundaries (e.g., Oromia, Amhara)
- `eth_admin2.shp` - Zone boundaries
- `eth_admin3.shp` - Woreda boundaries

You can find administrative boundary data from sources such as:
- [Humanitarian Data Exchange (HDX)](https://data.humdata.org/)
- [GADM](https://gadm.org/)
- [OpenStreetMap](https://www.openstreetmap.org/)

## How to Reproduce the Analysis

### Step 1: Download ACLED Data

Download ACLED conflict events for Ethiopia:

```bash
# Using Python script
python scripts/download_acled_data.py

# Or use the Jupyter notebook
jupyter notebook notebooks/01_download_acled.ipynb
```

This downloads ACLED events for Ethiopia from 2008 to 2024 (configurable in `src/config.py`) and saves raw data to `data/raw/`.

### Step 2: Prepare Data with Geographic Integration

Clean the data and integrate administrative boundaries:

```bash
# Process all years with Admin Level 1 (regions)
python src/data_prep.py --admin-level 1 --save-interim

# Or process a specific year
python src/data_prep.py --admin-level 1 --year 2020 --save-interim
```

This will:
- Clean and validate ACLED data
- Load administrative boundary shapefiles
- Perform spatial joins (point-in-polygon) to link events to regions
- Save cleaned data to `data/interim/`

### Step 3: Feature Engineering

Create analysis-ready features with optional period balancing:

```bash
# Standard feature engineering (no balancing)
python src/features.py --admin-level 1 --save-processed

# Time-based period balancing (RECOMMENDED)
# Preserves all post-Abiy data and matches pre-Abiy duration
python src/features.py --admin-level 1 --balance-periods --preserve-all-post --save-processed

# Equal time windows (e.g., 5 years each)
python src/features.py --admin-level 1 --balance-periods --years-per-period 5 --save-processed

# Equal event counts (with optional stratification)
python src/features.py --admin-level 1 --balance-periods --equal-counts --save-processed
python src/features.py --admin-level 1 --balance-periods --equal-counts --stratify-by adm1_name --save-processed
```

**Period Balancing Options:**
- `--preserve-all-post`: Preserves all post-Abiy events and sets pre-Abiy window to match post duration (time-based, not event-count-based)
- `--years-per-period N`: For equal time windows mode (default: 5)
- `--equal-counts`: Balance to equal number of events via downsampling
- `--stratify-by COL1 COL2`: Stratify sampling by columns (e.g., region, event type)

This creates processed data with:
- Pre/post Abiy period indicators
- Event type categorization
- Monthly and regional aggregations
- Saved to `data/processed/`

### Step 4: Run Analysis Modules

#### Time-Series Analysis

```bash
python src/analysis_time_series.py
```

Generates:
- Monthly event and fatality series
- Pre/post period summary statistics

#### Spatio-Temporal Analysis

```bash
# Analyze at Admin Level 1 (regions)
python src/analysis_spatiotemporal.py --admin-level 1

# Analyze at Admin Level 2 (zones)
python src/analysis_spatiotemporal.py --admin-level 2
```

Generates:
- Regional distribution of events
- Regional pre/post comparisons
- Change metrics by region

#### Statistical Analysis (Interrupted Time-Series Regression)

```bash
python src/analysis_statistical.py
```

Generates:
- ITS regression model results
- Pre/post comparison tables
- Exports results to CSV, Markdown, and PNG

### Step 5: Export All Tables

Export all analysis tables as publication-ready PNG images:

```bash
# Export all tables (default: Admin Level 1)
python src/export_all_tables.py

# Export tables for specific admin level
python src/export_all_tables.py --admin-level 1
```

This generates:
- `tbl01_pre_post_overall.csv/png` - Summary statistics
- `tbl02_event_type_distribution.csv/png` - Event type distribution
- `tbl03_regional_averages_admin1.csv/png` - Regional averages
- `tbl04_its_coefficients.csv/md/png` - ITS regression results

All tables are exported in three formats:
- **CSV**: For data analysis
- **Markdown**: For documentation
- **PNG**: Publication-ready images with formatting

### Step 6: Inspect Results

- **Tables**: Check `reports/tables/` for summary statistics and regression results
- **Figures**: Check `reports/figures/` for visualizations (when visualization module is implemented)

## Configuration

Key parameters can be adjusted in `src/config.py`:

- `ABIY_CUTOFF_DATE`: The date Abiy Ahmed was sworn in (default: 2018-04-02)
- `START_YEAR`: Beginning of analysis window (default: 2008, 10 years before cutoff)
- `END_YEAR`: End of analysis window (default: 2025)
- `COUNTRY`: Country name for ACLED queries (default: "Ethiopia")
- `ACLED_BASE_URL`: ACLED API endpoint
- `ACLED_TOKEN_URL`: OAuth token endpoint

## Period Balancing

The pipeline supports multiple period balancing strategies:

### 1. Time-Based Balancing (Recommended)

Preserves all post-Abiy data and matches pre-Abiy duration:

```bash
python src/features.py --balance-periods --preserve-all-post --save-processed
```

**Example:**
- Post-Abiy: 2018-04-02 to 2024-12-10 (6.69 years)
- Pre-Abiy: 2011-07-24 to 2018-04-02 (6.69 years) ✓

This ensures equal time periods for fair temporal comparison, regardless of event counts.

### 2. Equal Time Windows

Selects equal duration periods before and after the cutoff:

```bash
python src/features.py --balance-periods --years-per-period 5 --save-processed
```

**Example:**
- Pre-Abiy: 2013-04-02 to 2018-04-02 (5 years)
- Post-Abiy: 2018-04-02 to 2023-04-02 (5 years)

### 3. Equal Event Counts

Downsamples to equal number of events:

```bash
python src/features.py --balance-periods --equal-counts --save-processed
```

Optionally with stratified sampling:

```bash
python src/features.py --balance-periods --equal-counts --stratify-by adm1_name event_category --save-processed
```

## Outputs

### Tables

All tables are exported in multiple formats:

- **CSV** (`*.csv`): Raw data for analysis
- **Markdown** (`*.md`): Formatted for documentation
- **PNG** (`*.png`): Publication-ready images with:
  - Professional formatting
  - Titles and subtitles
  - Alternating row colors
  - High DPI (300 DPI)
  - Optimized spacing

**Table Files:**
- `tbl01_pre_post_overall.csv/md/png`: Summary of monthly events/fatalities
- `tbl02_event_type_distribution.csv/md/png`: Event type distribution pre vs post
- `tbl03_regional_averages_admin1.csv/md/png`: Regional averages pre vs post
- `tbl04_its_coefficients.csv/md/png`: Interrupted time-series regression coefficients

### Figures

(To be implemented in Phase 6: Visualization)

- `fig01_monthly_events.png`: Monthly conflict events with cutoff annotation
- `fig02_monthly_fatalities.png`: Monthly conflict fatalities with cutoff annotation
- `fig03_event_type_composition.png`: Event type distribution pre vs post
- `fig04_regional_distribution.png`: Regional distribution pre vs post (choropleths)
- `fig05_regional_change.png`: Change in conflict intensity by region
- `fig06_its_model_fit.png`: ITS model fit overlay

## Module Reference

### Core Modules

- **`src/acled_client.py`**: ACLED API client with OAuth authentication
  - Handles token acquisition and refresh
  - Fetches data with pagination (5000 records per page)
  
- **`src/data_prep.py`**: Data preparation and geographic integration
  - Cleans and validates ACLED data
  - Loads administrative boundaries
  - Performs spatial joins (point-in-polygon)
  
- **`src/features.py`**: Feature engineering
  - Adds pre/post Abiy period indicators
  - Categorizes event types
  - Creates aggregations (monthly, regional, event type)
  - Period balancing functionality

### Analysis Modules

- **`src/analysis_time_series.py`**: Temporal trend analysis
  - Monthly event/fatality series
  - Pre/post period comparisons
  
- **`src/analysis_spatiotemporal.py`**: Spatial and spatio-temporal analysis
  - Regional distribution
  - Regional pre/post comparisons
  - Change metrics
  
- **`src/analysis_statistical.py`**: Statistical analysis
  - Interrupted Time-Series (ITS) regression
  - Pre/post comparison tables
  - Model diagnostics

### Export Modules

- **`src/table_export.py`**: Publication-ready table export
  - PNG image generation with formatting
  - CSV and Markdown export
  
- **`src/export_all_tables.py`**: Batch table export
  - Orchestrates export of all required tables
  - Supports multiple admin levels

## Testing

Run basic validation tests:

```bash
pytest tests/
```

## Data Sources

- **ACLED**: [Armed Conflict Location & Event Data Project](https://acleddata.com/)
  - API Documentation: [https://acleddata.com/api-documentation/](https://acleddata.com/api-documentation/)
  - OAuth Authentication: [https://acleddata.com/api-documentation/getting-started](https://acleddata.com/api-documentation/getting-started)
- **Administrative Boundaries**: User-provided shapefiles (see Setup Step 5)

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:
1. Verify your `.env` file contains `ACLED_USERNAME` and `ACLED_PASSWORD`
2. Ensure your myACLED account credentials are correct
3. Check that your account has API access enabled

### Geographic Data Issues

If spatial joins fail:
1. Verify shapefiles are in `geo/ethiopia_admin_boundaries/`
2. Check that coordinate reference system is EPSG:4326 (WGS84)
3. Review `geo/ethiopia_admin_boundaries/README.md` for details

### Module Import Errors

If you encounter `ModuleNotFoundError`:
1. Ensure you're running from the project root directory
2. Activate the virtual environment: `source venv/bin/activate`
3. Verify all dependencies are installed: `pip install -r requirements.txt`

## License

[Specify your license here]

## Citation

If you use this analysis pipeline, please cite:
- ACLED data: [ACLED citation format]
- This analysis pipeline: [Your citation]

## Contact

[Your contact information]
