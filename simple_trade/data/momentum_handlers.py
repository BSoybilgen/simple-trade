"""
Handlers for momentum indicators like RSI, MACD, Stochastic, CCI, etc.
"""
import pandas as pd


def handle_stochastic(df, indicator_func, **indicator_kwargs):
    """Handle Stochastic Oscillator indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    k_period = kwargs_copy.pop('k_period', 14)  # Default k_period
    d_period = kwargs_copy.pop('d_period', 3)   # Default d_period
    smooth_k = kwargs_copy.pop('smooth_k', 3)   # Default smooth_k
    
    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name, close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}', '{low_col_name}', and '{close_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, 
                         k_period=k_period, d_period=d_period, 
                         smooth_k=smooth_k, 
                         high_col=high_col_name, 
                         low_col=low_col_name, 
                         close_col=close_col_name)


def handle_cci(df, indicator_func, **indicator_kwargs):
    """Handle Commodity Channel Index (CCI) indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 20)  # Default window
    constant = kwargs_copy.pop('constant', 0.015)  # Default constant
    
    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name, close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}', '{low_col_name}', and '{close_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, 
                         window=window, constant=constant, 
                         high_col=high_col_name, 
                         low_col=low_col_name, 
                         close_col=close_col_name)


def handle_roc(df, indicator_func, **indicator_kwargs):
    """Handle Rate of Change (ROC) indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 12)  # Default window

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' column."
        )
    
    # Calculate indicator
    return indicator_func(df, window=window, 
                         close_col=close_col_name)


def handle_macd(df, indicator_func, **indicator_kwargs):
    """Handle MACD indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window_fast = kwargs_copy.pop('window_fast', 12)
    window_slow = kwargs_copy.pop('window_slow', 26)
    window_signal = kwargs_copy.pop('window_signal', 9)

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' column."
        )
    
    # Calculate indicator
    return indicator_func(df, window_fast=window_fast, 
                         window_slow=window_slow, window_signal=window_signal, 
                         close_col=close_col_name)


def handle_rsi(df, indicator_func, **indicator_kwargs):
    """Handle RSI indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' column."
        )
    
    # Calculate indicator
    return indicator_func(df, window=window, close_col=close_col_name)
