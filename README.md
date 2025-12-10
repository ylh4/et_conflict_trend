# Conflict in Ethiopia: Before and After Abiy Ahmed

A comprehensive data analysis pipeline for analyzing conflict trends in Ethiopia before and after Abiy Ahmed's assumption of power (April 2, 2018), using ACLED (Armed Conflict Location & Event Data Project) data.

## Overview

This project implements a reproducible Python analysis stack that:
- Downloads and processes ACLED conflict data for Ethiopia
- Analyzes temporal trends (time-series analysis)
- Examines spatial patterns (spatio-temporal analysis)
- Performs statistical analysis (interrupted time-series regression)
- Generates publication-quality figures and tables for blog publication

## Project Structure

```
ethiopia_conflict_pre_post_abiy/
├── data/
│   ├── raw/              # Raw ACLED data (CSV files)
│   ├── interim/          # Cleaned data
│   └── processed/        # Feature-engineered data
├── geo/
│   └── ethiopia_admin_boundaries/  # Administrative boundary shapefiles/GeoJSON
├── notebooks/
│   ├── 01_download_acled.ipynb     # Data download notebook
│   └── 99_pipeline_run.ipynb       # End-to-end pipeline
├── src/
│   ├── config.py                    # Central configuration
│   ├── acled_client.py              # ACLED API client
│   ├── data_prep.py                 # Data cleaning and geographic loading
│   ├── features.py                  # Feature engineering
│   ├── analysis_time_series.py      # Time-series analysis
│   ├── analysis_spatiotemporal.py   # Spatial analysis
│   ├── analysis_statistical.py      # Statistical analysis (ITS)
│   ├── visualization.py             # Figure generation
│   └── utils_logging.py             # Logging utilities
├── scripts/                          # Standalone scripts
├── tests/                           # Unit tests
├── reports/
│   ├── figures/                     # Generated figures (PNG)
│   └── tables/                      # Generated tables (CSV, Markdown)
├── .env                             # ACLED API credentials (create from .env.example)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

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

1. Register for an ACLED API account at [https://developer.acleddata.com/](https://developer.acleddata.com/)
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your credentials:
   ```
   ACLED_EMAIL=your_registered_email@example.org
   ACLED_API_KEY=your_acled_api_key_here
   ```

**Important:** Never commit the `.env` file to version control. It is already included in `.gitignore`.

### 5. Prepare Geographic Data

Download Ethiopia administrative boundary data (regions, zones, woredas) and place the shapefiles or GeoJSON files in:
```
geo/ethiopia_admin_boundaries/
```

You can find administrative boundary data from sources such as:
- [Humanitarian Data Exchange (HDX)](https://data.humdata.org/)
- [GADM](https://gadm.org/)
- [OpenStreetMap](https://www.openstreetmap.org/)

## How to Reproduce the Analysis

### Step 1: Download ACLED Data

Run the data download script or notebook:

```bash
# Using Python script (when available)
python scripts/download_acled_data.py

# Or use the Jupyter notebook
jupyter notebook notebooks/01_download_acled.ipynb
```

This will download ACLED events for Ethiopia from 2013 to 2025 (configurable in `src/config.py`) and save raw data to `data/raw/`.

### Step 2: Run Data Cleaning & Feature Engineering

The cleaning and feature engineering steps are typically run as part of the main pipeline (see Step 4).

### Step 3: Prepare Geographic Data

Ensure administrative boundary shapefiles/GeoJSON files are in `geo/ethiopia_admin_boundaries/`. The pipeline will automatically load them when needed.

### Step 4: Run the Complete Pipeline

Execute the end-to-end analysis pipeline:

```bash
# Using Python script (when available)
python scripts/run_pipeline.py

# Or use the Jupyter notebook
jupyter notebook notebooks/99_pipeline_run.ipynb
```

The pipeline will:
1. Load and clean ACLED data
2. Engineer features (pre/post Abiy classification, aggregations)
3. Perform time-series analysis
4. Conduct spatio-temporal analysis
5. Run statistical analysis (interrupted time-series regression)
6. Generate all figures and tables
7. Save outputs to `reports/figures/` and `reports/tables/`

### Step 5: Inspect Results

- **Figures**: Check `reports/figures/` for publication-ready visualizations
- **Tables**: Check `reports/tables/` for summary statistics and regression results

## Configuration

Key parameters can be adjusted in `src/config.py`:

- `ABIY_CUTOFF_DATE`: The date Abiy Ahmed was sworn in (default: 2018-04-02)
- `START_YEAR`: Beginning of analysis window (default: 2013, 5 years before cutoff)
- `END_YEAR`: End of analysis window (default: 2025)
- `COUNTRY`: Country name for ACLED queries (default: "Ethiopia")

## Outputs

### Figures

- `fig01_monthly_events.png`: Monthly conflict events with cutoff annotation
- `fig02_monthly_fatalities.png`: Monthly conflict fatalities with cutoff annotation
- `fig03_event_type_composition.png`: Event type distribution pre vs post
- `fig04_regional_distribution.png`: Regional distribution pre vs post (choropleths)
- `fig05_regional_change.png`: Change in conflict intensity by region
- `fig06_its_model_fit.png`: ITS model fit overlay (optional)

### Tables

- `tbl01_pre_post_overall.csv/md`: Summary of monthly events/fatalities
- `tbl02_event_type_distribution.csv/md`: Event type distribution pre vs post
- `tbl03_regional_averages.csv/md`: Regional averages pre vs post
- `tbl04_its_coefficients.csv/md`: Interrupted time-series regression coefficients

## Testing

Run basic validation tests:

```bash
pytest tests/
```

## Data Sources

- **ACLED**: [Armed Conflict Location & Event Data Project](https://acleddata.com/)
- **Administrative Boundaries**: User-provided shapefiles/GeoJSON (see Setup Step 5)

## License

[Specify your license here]

## Citation

If you use this analysis pipeline, please cite:
- ACLED data: [ACLED citation format]
- This analysis pipeline: [Your citation]

## Contact

[Your contact information]

