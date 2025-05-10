"""
Handlers for volume indicators like OBV, etc.
"""
import pandas as pd


def handle_obv(df, indicator_func, **indicator_kwargs):
    """Handle On-Balance Volume (OBV) indicator calculation."""
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')
    volume_col_name = kwargs_copy.pop('volume_col', 'Volume')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name, volume_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' and '{volume_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, close_col=close_col_name, volume_col=volume_col_name)


def handle_vma(df, indicator_func, **indicator_kwargs):
    """Handle Volume Moving Average (VMA) indicator calculation."""
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window
    
    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')
    volume_col_name = kwargs_copy.pop('volume_col', 'Volume')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name, volume_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' and '{volume_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, window=window, close_col=close_col_name, volume_col=volume_col_name)


def handle_adline(df, indicator_func, **indicator_kwargs):
    """Handle Accumulation/Distribution Line (A/D Line) indicator calculation."""
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()
    
    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')
    close_col_name = kwargs_copy.pop('close_col', 'Close')
    volume_col_name = kwargs_copy.pop('volume_col', 'Volume')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name, close_col_name, volume_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}', '{low_col_name}', '{close_col_name}' and '{volume_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, high_col=high_col_name, low_col=low_col_name, close_col=close_col_name, volume_col=volume_col_name)


def handle_cmf(df, indicator_func, **indicator_kwargs):
    """Handle Chaikin Money Flow (CMF) indicator calculation."""
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    period = kwargs_copy.pop('period', 20)  # Default period
    
    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')
    close_col_name = kwargs_copy.pop('close_col', 'Close')
    volume_col_name = kwargs_copy.pop('volume_col', 'Volume')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name, close_col_name, volume_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}', '{low_col_name}', '{close_col_name}' and '{volume_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, period=period, high_col=high_col_name, low_col=low_col_name, close_col=close_col_name, volume_col=volume_col_name)


def handle_vpt(df, indicator_func, **indicator_kwargs):
    """Handle Volume Price Trend (VPT) indicator calculation."""
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')
    volume_col_name = kwargs_copy.pop('volume_col', 'Volume')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name, volume_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' and '{volume_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, close_col=close_col_name, volume_col=volume_col_name)