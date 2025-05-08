"""
Handlers for trend indicators like SMA, EMA, ADX, etc.
"""
import pandas as pd


def handle_strend(df, indicator_func, **indicator_kwargs):
    """Handle Supertrend indicator calculation."""

    kwargs_copy = indicator_kwargs.copy()

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
    
    # Extract actual parameters for the Supertrend function
    period = kwargs_copy.pop('period', 7)
    multiplier = kwargs_copy.pop('multiplier', 3.0)
    
    # Call the indicator_func (core supertrend function)
    # It expects the DataFrame and then the string names for columns, plus parameters.
    # Pass any remaining kwargs_copy as well.
    return indicator_func(
        df,  # Pass the original DataFrame
        period=period, 
        multiplier=multiplier,
        high_col=high_col_name,    # Pass the string name for the high column
        low_col=low_col_name,      # Pass the string name for the low column
        close_col=close_col_name  # Pass the string name for the close column
    )


def handle_adx(df, indicator_func, **indicator_kwargs):
    """Handle ADX indicator calculation."""
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default ADX window
    
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
    return indicator_func(
        df,
        window=window, 
        high_col=high_col_name, 
        low_col=low_col_name, 
        close_col=close_col_name)


def handle_psar(df, indicator_func, **indicator_kwargs):
    """Handle Parabolic SAR indicator calculation."""
        
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    af_initial = kwargs_copy.pop('af_initial', 0.02)
    af_step = kwargs_copy.pop('af_step', 0.02)
    af_max = kwargs_copy.pop('af_max', 0.2)

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
    return indicator_func(
        df,
        af_initial=af_initial, 
        af_step=af_step, 
        af_max=af_max,
        high_col=high_col_name,    # Pass the string name for the high column
        low_col=low_col_name,      # Pass the string name for the low column
        close_col=close_col_name   # Pass the string name for the close column)
    )

def handle_ichimoku(df, indicator_func, indicator, **indicator_kwargs):
    """Handle Ichimoku Cloud and its components."""
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    tenkan_period = kwargs_copy.pop('tenkan_period', 9)  # Default tenkan period
    kijun_period = kwargs_copy.pop('kijun_period', 26)  # Default kijun period
    senkou_b_period = kwargs_copy.pop('senkou_b_period', 52)  # Default senkou span B period
    displacement = kwargs_copy.pop('displacement', 26)  # Default displacement
    
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
    
    # Calculate full Ichimoku
    return indicator_func(df,
        tenkan_period=tenkan_period, 
        kijun_period=kijun_period, 
        senkou_b_period=senkou_b_period, 
        displacement=displacement,
        high_col=high_col_name, 
        low_col=low_col_name, 
        close_col=close_col_name)

def handle_aroon(df, indicator_func, **indicator_kwargs):
    """Handle Aroon indicator calculation."""
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    period = kwargs_copy.pop('period', 14)  # Default period
    
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
    return indicator_func(df, period=period, high_col=high_col_name, low_col=low_col_name)
