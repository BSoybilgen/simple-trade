"""
Main indicator handling module that coordinates the calculation of various technical indicators.
"""
import yfinance as yf
import pandas as pd
from .core import INDICATORS


def compute_indicator(
    data: pd.DataFrame,
    indicator: str,
    **indicator_kwargs
) -> pd.DataFrame:
    """Computes a specified technical indicator on the provided financial data.

    Args:
        data: pandas.DataFrame containing the financial data (must include 'Close',
              and possibly 'High', 'Low' depending on the indicator).
        indicator: Technical indicator to compute (e.g., 'rsi', 'sma', 'macd', 'adx').
        **indicator_kwargs: Keyword arguments specific to the chosen indicator.

    Returns:
        pandas.DataFrame: Original DataFrame with the calculated indicator column(s) added.

    Raises:
        ValueError: If the indicator is not supported or the required columns are missing.
    """
    # Validate indicator exists
    if indicator not in INDICATORS:
        raise ValueError(f"Indicator '{indicator}' not supported. Available: {list(INDICATORS.keys())}")

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    indicator_func = INDICATORS[indicator]
    print(f"Computing {indicator.upper()}...")

    try:
        # Delegate to specific handler based on indicator type
        indicator_result = _calculate_indicator(df, indicator_func, **indicator_kwargs)
        
        # Add the result to the original DataFrame
        df = _add_indicator_to_dataframe(df, indicator_result, indicator_kwargs)
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicator '{indicator}': {e}")
        return df  # Return the original df if calculation fails


def _calculate_indicator(df, indicator_func, **indicator_kwargs):
    """Dispatch to the appropriate handler for each indicator type."""
    # Keep parameters and columns as dictionaries
    # This is important for indicator functions that expect them as dictionaries
    return indicator_func(df, **indicator_kwargs)


def _add_indicator_to_dataframe(df, indicator_result, indicator_kwargs):
    """Add the calculated indicator to the DataFrame with appropriate naming."""
    if isinstance(indicator_result, pd.Series):
        df[indicator_result.name] = indicator_result
        
    elif isinstance(indicator_result, pd.DataFrame):
        df = df.join(indicator_result)
        
    else:
        print(f"Warning: Indicator function returned an unexpected type: {type(indicator_result)}")
    
    return df


def download_data(symbol: str, start_date: str, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
    """Download historical price data for a given symbol using yfinance."""
    # Set auto_adjust=False to get raw OHLCV and prevent yfinance from potentially altering columns
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}.")

    # Clean up column names: remove multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        # Remove duplicate columns
        df = df.loc[:,~df.columns.duplicated()]

    # Force column names to lowercase for consistent mapping
    df.columns = df.columns.str.lower()

    # Standardize column names to Title Case
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Adj Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_map)

    # Ensure all expected columns are present, derived if needed
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']  # Use Close as Adj Close if not available

    # Add a symbol attribute to the dataframe for reference
    df.attrs['symbol'] = symbol

    return df