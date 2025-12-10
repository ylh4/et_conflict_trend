"""
ACLED API client for downloading conflict event data.

This module provides functions to interact with the ACLED API to download
conflict events for Ethiopia, handling OAuth authentication, pagination, and caching raw data.

The ACLED API now uses OAuth token-based authentication. This client automatically
handles token acquisition and refresh.
"""

import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timedelta

from src.config import (
    ACLED_BASE_URL,
    ACLED_TOKEN_URL,
    ACLED_USERNAME,
    ACLED_PASSWORD,
    COUNTRY,
    RAW_DATA_DIR,
    validate_credentials,
)
from src.utils_logging import get_logger

logger = get_logger(__name__)

# Rate limiting: be polite to the API
REQUEST_DELAY = 0.5  # seconds between requests

# Token cache (in-memory)
_token_cache: Optional[Dict] = None
_token_expires_at: Optional[datetime] = None


def get_access_token(force_refresh: bool = False) -> str:
    """
    Get an OAuth access token from ACLED API.
    
    Uses cached token if still valid. Otherwise, requests a new token.
    Access tokens expire in 24 hours; refresh tokens are valid for 14 days.
    
    Args:
        force_refresh: If True, force a new token request even if cached token exists.
    
    Returns:
        Access token string.
    
    Raises:
        requests.HTTPError: If token request fails.
        ValueError: If credentials are missing or response is invalid.
    """
    global _token_cache, _token_expires_at
    
    # Check if we have a valid cached token
    if not force_refresh and _token_cache and _token_expires_at:
        if datetime.now() < _token_expires_at:
            logger.debug("Using cached access token")
            return _token_cache["access_token"]
        else:
            logger.info("Cached token expired, requesting new token")
    
    # Validate credentials
    validate_credentials()
    
    # Request new token
    logger.info("Requesting OAuth access token from ACLED API...")
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {
        "username": ACLED_USERNAME,
        "password": ACLED_PASSWORD,
        "grant_type": "password",
        "client_id": "acled",
    }
    
    try:
        response = requests.post(ACLED_TOKEN_URL, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Validate response
        if "access_token" not in token_data:
            raise ValueError(f"Invalid token response: missing 'access_token'. Response: {token_data}")
        
        # Cache the token
        _token_cache = token_data
        # Set expiration time (24 hours, with 5 minute buffer)
        expires_in = token_data.get("expires_in", 86400)  # Default 24 hours
        _token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)  # 5 min buffer
        
        logger.info("Successfully obtained access token")
        return token_data["access_token"]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get access token: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise
    except ValueError as e:
        logger.error(f"Invalid token response: {e}")
        raise


def refresh_access_token() -> str:
    """
    Refresh the access token using the refresh token.
    
    Returns:
        New access token string.
    
    Raises:
        ValueError: If no refresh token is available or refresh fails.
    """
    global _token_cache, _token_expires_at
    
    if not _token_cache or "refresh_token" not in _token_cache:
        logger.warning("No refresh token available, requesting new token with credentials")
        return get_access_token(force_refresh=True)
    
    logger.info("Refreshing access token...")
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {
        "refresh_token": _token_cache["refresh_token"],
        "grant_type": "refresh_token",
        "client_id": "acled",
    }
    
    try:
        response = requests.post(ACLED_TOKEN_URL, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        
        if "access_token" not in token_data:
            raise ValueError(f"Invalid refresh response: missing 'access_token'. Response: {token_data}")
        
        # Update cache
        _token_cache = token_data
        expires_in = token_data.get("expires_in", 86400)
        _token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)
        
        logger.info("Successfully refreshed access token")
        return token_data["access_token"]
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Token refresh failed: {e}. Requesting new token with credentials.")
        return get_access_token(force_refresh=True)


def fetch_acled_page(params: Dict, page: int = 1, retry_on_auth_error: bool = True) -> Dict:
    """
    Fetch a single page of ACLED data from the API.
    
    Args:
        params: Dictionary of query parameters (country, year, limit, etc.)
        page: Page number to fetch (default: 1)
        retry_on_auth_error: If True, retry once with a fresh token on 401/403 errors.
    
    Returns:
        Dictionary containing the API response with 'data' and 'count' keys.
    
    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If the response format is unexpected.
    """
    # Get access token
    access_token = get_access_token()
    
    # Add pagination and format parameters
    request_params = params.copy()
    request_params["page"] = page
    request_params["_format"] = "json"
    
    # Prepare headers with OAuth token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get(ACLED_BASE_URL, params=request_params, headers=headers, timeout=30)
        
        # Handle authentication errors
        if response.status_code in (401, 403) and retry_on_auth_error:
            logger.warning(f"Authentication error (status {response.status_code}), refreshing token and retrying...")
            access_token = refresh_access_token()
            headers["Authorization"] = f"Bearer {access_token}"
            response = requests.get(ACLED_BASE_URL, params=request_params, headers=headers, timeout=30)
        
        response.raise_for_status()
        
        data = response.json()
        
        # Validate response structure
        if "status" in data and data["status"] != 200:
            raise ValueError(f"API returned error status {data['status']}: {data.get('message', 'Unknown error')}")
        
        if "data" not in data:
            raise ValueError(f"Unexpected API response format: missing 'data' key. Response keys: {list(data.keys())}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for page {page}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise
    except ValueError as e:
        logger.error(f"Invalid response format for page {page}: {e}")
        raise


def fetch_acled_for_year(year: int, limit: int = 5000) -> pd.DataFrame:
    """
    Fetch all ACLED records for Ethiopia for a given year, handling pagination.
    
    Args:
        year: Year to fetch data for (e.g., 2018)
        limit: Number of records per page (default: 5000, max 5000 per ACLED API)
    
    Returns:
        DataFrame containing all ACLED events for Ethiopia in the specified year.
    
    Raises:
        ValueError: If credentials are missing or API request fails.
    """
    # Validate credentials
    validate_credentials()
    
    # Build base query parameters (no key/email needed with OAuth)
    params = {
        "country": COUNTRY,
        "year": year,
        "limit": limit,
    }
    
    logger.info(f"Fetching ACLED data for {COUNTRY}, year {year}")
    
    all_records = []
    page = 1
    total_count = None
    
    # Fetch first page to get total count
    try:
        first_response = fetch_acled_page(params, page=1)
        all_records.extend(first_response["data"])
        total_count = first_response.get("count", len(first_response["data"]))
        
        logger.info(f"Total records for {year}: {total_count}")
        
        # Calculate number of pages needed
        records_per_page = len(first_response["data"])
        if records_per_page == 0:
            logger.warning(f"No records found for {COUNTRY} in year {year}")
            return pd.DataFrame()
        
        total_pages = (total_count + records_per_page - 1) // records_per_page
        
        # Fetch remaining pages
        if total_pages > 1:
            logger.info(f"Fetching {total_pages} pages of data...")
            for page_num in tqdm(range(2, total_pages + 1), desc=f"Year {year}"):
                time.sleep(REQUEST_DELAY)  # Rate limiting
                page_response = fetch_acled_page(params, page=page_num)
                all_records.extend(page_response["data"])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        logger.info(f"Successfully fetched {len(df)} records for year {year}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for year {year}: {e}")
        raise


def save_raw(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save raw ACLED data to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the CSV file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} records to {filepath}")


def fetch_acled_range(start_year: int, end_year: int, save_individual: bool = True) -> pd.DataFrame:
    """
    Fetch ACLED data for Ethiopia for a range of years.
    
    Args:
        start_year: First year to fetch (inclusive)
        end_year: Last year to fetch (inclusive)
        save_individual: If True, save each year's data as separate CSV file
    
    Returns:
        DataFrame containing all ACLED events for the specified year range.
    """
    validate_credentials()
    
    logger.info(f"Fetching ACLED data for {COUNTRY} from {start_year} to {end_year}")
    
    all_dataframes = []
    
    for year in range(start_year, end_year + 1):
        try:
            # Fetch data for this year
            df_year = fetch_acled_for_year(year)
            
            if not df_year.empty:
                # Save individual year file if requested
                if save_individual:
                    year_filepath = RAW_DATA_DIR / f"acled_ethiopia_{year}.csv"
                    save_raw(df_year, year_filepath)
                
                all_dataframes.append(df_year)
                logger.info(f"Year {year}: {len(df_year)} records")
            else:
                logger.warning(f"No data found for year {year}")
            
            # Small delay between years to be polite to the API
            if year < end_year:
                time.sleep(REQUEST_DELAY)
                
        except Exception as e:
            logger.error(f"Failed to fetch data for year {year}: {e}")
            logger.warning(f"Continuing with other years...")
            continue
    
    # Combine all years
    if all_dataframes:
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {len(df_combined)} total records")
        
        # Save combined file
        combined_filepath = RAW_DATA_DIR / "acled_ethiopia_all_years.csv"
        save_raw(df_combined, combined_filepath)
        
        return df_combined
    else:
        logger.warning("No data was successfully fetched")
        return pd.DataFrame()


def load_cached_data(year: Optional[int] = None) -> pd.DataFrame:
    """
    Load cached ACLED data from disk.
    
    Args:
        year: If specified, load data for a specific year. Otherwise, load combined file.
    
    Returns:
        DataFrame with cached data, or empty DataFrame if file doesn't exist.
    """
    if year:
        filepath = RAW_DATA_DIR / f"acled_ethiopia_{year}.csv"
    else:
        filepath = RAW_DATA_DIR / "acled_ethiopia_all_years.csv"
    
    if filepath.exists():
        logger.info(f"Loading cached data from {filepath}")
        return pd.read_csv(filepath)
    else:
        logger.warning(f"No cached data found at {filepath}")
        return pd.DataFrame()
