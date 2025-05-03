"""
Handlers for volatility indicators like Bollinger Bands, ATR, Keltner Channels, etc.
"""
import pandas as pd


def handle_bollin(df, indicator_func, **indicator_kwargs):
    """Handle Bollinger Bands indicator calculation."""
    # Validate required columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column for Bollinger Bands calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    if 'std_dev' in kwargs_copy:
        # Rename std_dev to num_std for our local bollinger_bands function
        kwargs_copy['num_std'] = kwargs_copy.pop('std_dev') 
    else:
        # Default num_std if std_dev wasn't provided
        kwargs_copy['num_std'] = 2 
    
    # Ensure 'window' is also present, default to 20
    if 'window' not in kwargs_copy:
        kwargs_copy['window'] = 20
    
    # Calculate indicator
    return indicator_func(df['Close'], **kwargs_copy)


def handle_atr(df, indicator_func, **indicator_kwargs):
    """Handle Average True Range (ATR) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for ATR calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], 
                         window=window, **kwargs_copy)


def handle_kelt(df, indicator_func, **indicator_kwargs):
    """Handle Keltner Channels indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for Keltner Channels calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    ema_window = kwargs_copy.pop('ema_window', 20)  # Default EMA window
    atr_window = kwargs_copy.pop('atr_window', 10)  # Default ATR window
    atr_multiplier = kwargs_copy.pop('atr_multiplier', 2.0)  # Default multiplier
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], 
                        ema_window=ema_window, atr_window=atr_window,
                        atr_multiplier=atr_multiplier, **kwargs_copy)


def handle_donch(df, indicator_func, **indicator_kwargs):
    """Handle Donchian Channels indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns for Donchian Channels calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 20)  # Default window
    
    # Remove H/L string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], window=window, **kwargs_copy)


def handle_chaik(df, indicator_func, **indicator_kwargs):
    """Handle Chaikin Volatility indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns for Chaikin Volatility calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    ema_window = kwargs_copy.pop('ema_window', 10)  # Default EMA window
    roc_window = kwargs_copy.pop('roc_window', 10)  # Default ROC window
    
    # Remove H/L string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], 
                         ema_window=ema_window, 
                         roc_window=roc_window, 
                         **kwargs_copy)


def format_volatility_indicator_name(indicator, kwargs):
    """Format the display name for volatility indicators."""
    window_str = ""
    if indicator == 'bollin':
        # Just use the window param for naming
        window = kwargs.get('window', 20)
        window_str = f"_{window}"
    elif indicator == 'atr':
        # For ATR, show window in the name
        window = kwargs.get('window', 14)
        window_str = f"_{window}"
    elif indicator == 'kelt':
        # For Keltner Channels, show EMA window in the name
        ema_window = kwargs.get('ema_window', 20)
        window_str = f"_{ema_window}"
    elif indicator == 'donch':
        # For Donchian Channels, show window in the name
        window = kwargs.get('window', 20)
        window_str = f"_{window}"
    elif indicator == 'chaik':
        # For Chaikin Volatility, show both windows in the name
        ema_window = kwargs.get('ema_window', 10)
        roc_window = kwargs.get('roc_window', 10)
        window_str = f"_{ema_window}_{roc_window}"
    
    return window_str
