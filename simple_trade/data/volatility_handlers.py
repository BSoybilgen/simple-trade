"""
Handlers for volatility indicators like Bollinger Bands, ATR, Keltner Channels, etc.
"""
import pandas as pd


def handle_bollin(df, indicator_func, **indicator_kwargs):
    """Handle Bollinger Bands indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    num_std = kwargs_copy.pop('std_dev', 2)
    window = kwargs_copy.pop('window', 20)

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    close_col_name = kwargs_copy.pop('close_col', 'Close')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [close_col_name]):
        raise ValueError(
            f"DataFrame must contain '{close_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df,
                          window=window, 
                          num_std=num_std, 
                          close_col=close_col_name)


def handle_atr(df, indicator_func, **indicator_kwargs):
    """Handle Average True Range (ATR) indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default window

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
                         window=window, 
                         high_col=high_col_name, 
                         low_col=low_col_name, 
                         close_col=close_col_name)


def handle_kelt(df, indicator_func, **indicator_kwargs):
    """Handle Keltner Channels indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    ema_window = kwargs_copy.pop('ema_window', 20)  # Default EMA window
    atr_window = kwargs_copy.pop('atr_window', 10)  # Default ATR window
    atr_multiplier = kwargs_copy.pop('atr_multiplier', 2.0)  # Default multiplier
    
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
                        ema_window=ema_window, atr_window=atr_window,
                        atr_multiplier=atr_multiplier, 
                        high_col=high_col_name, 
                        low_col=low_col_name, 
                        close_col=close_col_name)


def handle_donch(df, indicator_func, **indicator_kwargs):
    """Handle Donchian Channels indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 20)  # Default window

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}' and '{low_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, window=window, 
                         high_col=high_col_name, 
                         low_col=low_col_name)


def handle_chaik(df, indicator_func, **indicator_kwargs):
    """Handle Chaikin Volatility indicator calculation."""

    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    ema_window = kwargs_copy.pop('ema_window', 10)  # Default EMA window
    roc_window = kwargs_copy.pop('roc_window', 10)  # Default ROC window

    # Get column names for HLC from kwargs, defaulting to standard names
    # These will be passed to the core function, which expects string names
    high_col_name = kwargs_copy.pop('high_col', 'High')
    low_col_name = kwargs_copy.pop('low_col', 'Low')

    # Validate that the original DataFrame contains these columns
    if not all(col in df.columns for col in [high_col_name, low_col_name]):
        raise ValueError(
            f"DataFrame must contain '{high_col_name}' and '{low_col_name}' columns."
        )
    
    # Calculate indicator
    return indicator_func(df, 
                         ema_window=ema_window, 
                         roc_window=roc_window, 
                         high_col=high_col_name, 
                         low_col=low_col_name)
