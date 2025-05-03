"""
Handlers for volume indicators like OBV, etc.
"""
import pandas as pd


def handle_obv(df, indicator_func, **indicator_kwargs):
    """Handle On-Balance Volume (OBV) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['Close', 'Volume']):
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns for OBV calculation.")
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()
    kwargs_copy.pop('close', None)
    kwargs_copy.pop('volume', None)
    
    # Calculate indicator
    return indicator_func(df['Close'], df['Volume'], **kwargs_copy)


def handle_vma(df, indicator_func, **indicator_kwargs):
    """Handle Volume Moving Average (VMA) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['Close', 'Volume']):
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns for VMA calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window
    
    # Remove column names from kwargs if they were passed
    kwargs_copy.pop('close', None)
    kwargs_copy.pop('volume', None)
    
    # Calculate indicator
    return indicator_func(df['Close'], df['Volume'], window=window, **kwargs_copy)


def handle_adline(df, indicator_func, **indicator_kwargs):
    """Handle Accumulation/Distribution Line (A/D Line) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns for A/D Line calculation.")
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    kwargs_copy.pop('volume', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], df['Volume'], **kwargs_copy)


def handle_cmf(df, indicator_func, **indicator_kwargs):
    """Handle Chaikin Money Flow (CMF) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns for CMF calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    period = kwargs_copy.pop('period', 20)  # Default period
    
    # Remove column names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    kwargs_copy.pop('volume', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], df['Volume'], period=period, **kwargs_copy)


def handle_vpt(df, indicator_func, **indicator_kwargs):
    """Handle Volume Price Trend (VPT) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['Close', 'Volume']):
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns for VPT calculation.")
    
    # Remove column names from kwargs if they were passed
    kwargs_copy = indicator_kwargs.copy()
    kwargs_copy.pop('close', None)
    kwargs_copy.pop('volume', None)
    
    # Calculate indicator
    return indicator_func(df['Close'], df['Volume'], **kwargs_copy)


def format_volume_indicator_name(indicator, kwargs):
    """Format the display name for volume indicators."""
    window_str = ""
    if indicator == 'obv':
        # OBV doesn't use parameters in the name
        return ""
    elif indicator == 'vma':
        # For VMA, show window in the name
        window = kwargs.get('window', 14)
        window_str = f"_{window}"
    elif indicator == 'adline':
        # A/D Line doesn't use parameters in the name
        return ""
    elif indicator == 'cmf':
        # For CMF, show period in the name
        period = kwargs.get('period', 20)
        window_str = f"_{period}"
    elif indicator == 'vpt':
        # VPT doesn't use parameters in the name
        return ""
    
    return window_str
