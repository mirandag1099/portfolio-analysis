import pandas as pd
import re

def load_portfolio(csv_path):
    """
    Load a portfolio CSV with flexible column names and weight formats.
    
    Handles:
    - Case-insensitive column names (ticker/Ticker/SYMBOL, weight/Weight/Allocation)
    - Weights as percentages (3.66%) or decimals (0.15)
    - BOM characters, whitespace, empty rows
    - Duplicate tickers (aggregates weights)
    - Missing data (drops rows with missing tickers)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with columns: ticker, weight (normalized to sum to 1)
    """
    # Try multiple encodings to handle BOM and special characters
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"Could not read CSV file with any supported encoding: {encodings}")
    
    # Clean column names: strip whitespace, remove BOM characters
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
    
    # Map flexible column names to standard names (case-insensitive)
    column_mapping = {}
    ticker_variants = ['ticker', 'symbol', 'stock', 'ticker_symbol', 'ticker symbol']
    weight_variants = ['weight', 'allocation', 'percentage', 'percent', 'pct', 'weight_pct', 'weight %']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        # Check for ticker column
        if any(variant in col_lower for variant in ticker_variants):
            column_mapping[col] = 'ticker'
        # Check for weight column
        elif any(variant in col_lower for variant in weight_variants):
            column_mapping[col] = 'weight'
    
    # Rename columns
    if 'ticker' not in column_mapping.values():
        raise ValueError(
            f"Could not find ticker column. Found columns: {list(df.columns)}. "
            f"Expected one of: {ticker_variants}"
        )
    if 'weight' not in column_mapping.values():
        raise ValueError(
            f"Could not find weight column. Found columns: {list(df.columns)}. "
            f"Expected one of: {weight_variants}"
        )
    
    df = df.rename(columns=column_mapping)
    
    # Keep only ticker and weight columns
    df = df[['ticker', 'weight']].copy()
    
    # Drop rows with missing tickers
    df = df.dropna(subset=['ticker'])
    
    # Clean ticker column: uppercase, strip whitespace, remove empty strings
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df[df['ticker'] != '']
    df = df[df['ticker'] != 'NAN']
    
    if df.empty:
        raise ValueError("No valid tickers found in CSV file.")
    
    # Clean weight column: handle percentages and decimals
    def parse_weight(value):
        """Convert weight to decimal (handles percentages and decimals)."""
        if pd.isna(value):
            return None
        
        # Convert to string and clean
        value_str = str(value).strip().replace(',', '')
        
        # Remove percentage sign if present
        is_percentage = '%' in value_str
        value_str = value_str.replace('%', '').strip()
        
        try:
            weight = float(value_str)
            # If it was a percentage, convert to decimal (e.g., 3.66% -> 0.0366)
            if is_percentage:
                weight = weight / 100.0
            return weight
        except (ValueError, TypeError):
            return None
    
    df['weight'] = df['weight'].apply(parse_weight)
    
    # Drop rows with invalid weights
    df = df.dropna(subset=['weight'])
    
    # Filter out zero or negative weights
    df = df[df['weight'] > 0]
    
    if df.empty:
        raise ValueError("No valid weights found in CSV file.")
    
    # Handle duplicate tickers: aggregate weights
    if df['ticker'].duplicated().any():
        df = df.groupby('ticker', as_index=False)['weight'].sum()
    
    # Validate total weight
    total_weight = df['weight'].sum()
    if total_weight <= 0:
        raise ValueError("Total weight must be greater than zero.")
    
    # Normalize weights to sum to 1
    if abs(total_weight - 1.0) > 1e-6:
        df['weight'] = df['weight'] / total_weight
    
    # Reset index and return
    df = df.reset_index(drop=True)
    
    return df
