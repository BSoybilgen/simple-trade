"""
Handlers for momentum indicators like RSI, MACD, Stochastic, CCI, etc.
"""
import pandas as pd


def handle_stochastic(df, indicator_func, **indicator_kwargs):
    """Handle Stochastic Oscillator indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for Stochastic Oscillator calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    k_period = kwargs_copy.pop('k_period', 14)  # Default k_period
    d_period = kwargs_copy.pop('d_period', 3)   # Default d_period
    smooth_k = kwargs_copy.pop('smooth_k', 3)   # Default smooth_k
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], 
                         k_period=k_period, d_period=d_period, 
                         smooth_k=smooth_k, **kwargs_copy)


def handle_cci(df, indicator_func, **indicator_kwargs):
    """Handle Commodity Channel Index (CCI) indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for CCI calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 20)  # Default window
    constant = kwargs_copy.pop('constant', 0.015)  # Default constant
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], 
                         window=window, constant=constant, **kwargs_copy)


def handle_roc(df, indicator_func, **indicator_kwargs):
    """Handle Rate of Change (ROC) indicator calculation."""
    # Validate required columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column for ROC calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 12)  # Default window
    
    # Calculate indicator
    return indicator_func(df['Close'], window=window, **kwargs_copy)


def handle_macd(df, indicator_func, **indicator_kwargs):
    """Handle MACD indicator calculation."""
    # Validate required columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column for MACD calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window_fast = kwargs_copy.pop('window_fast', 12)
    window_slow = kwargs_copy.pop('window_slow', 26)
    window_signal = kwargs_copy.pop('window_signal', 9)
    
    # Calculate indicator
    return indicator_func(df['Close'], window_fast=window_fast, 
                         window_slow=window_slow, window_signal=window_signal, 
                         **kwargs_copy)


def handle_rsi(df, indicator_func, **indicator_kwargs):
    """Handle RSI indicator calculation."""
    # Validate required columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column for RSI calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window
    
    # Calculate indicator
    return indicator_func(df['Close'], window=window, **kwargs_copy)


def format_momentum_indicator_name(indicator, kwargs):
    """Format the display name for momentum indicators."""
    window_str = ""
    if indicator == 'rsi':
        window = kwargs.get('window', 14)
        window_str = f"_{window}"
    elif indicator == 'macd':
        # MACD typically doesn't need a window suffix as it returns a DataFrame
        pass
    elif indicator == 'stoch':
        # For Stochastic Oscillator, show k_period in the name
        k_period = kwargs.get('k_period', 14)
        window_str = f"_{k_period}"
    elif indicator == 'cci':
        # For CCI, show window in the name
        window = kwargs.get('window', 20)
        window_str = f"_{window}"
    elif indicator == 'roc':
        # For ROC, show window in the name
        window = kwargs.get('window', 12)
        window_str = f"_{window}"
    
    return window_str
