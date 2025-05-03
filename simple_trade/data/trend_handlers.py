"""
Handlers for trend indicators like SMA, EMA, ADX, etc.
"""
import pandas as pd


def handle_strend(df, indicator_func, **indicator_kwargs):
    """Handle Supertrend indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for Supertrend calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    period = kwargs_copy.pop('period', 10)  # Default period
    multiplier = kwargs_copy.pop('multiplier', 3.0)  # Default multiplier
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)

    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], 
                        period=period, multiplier=multiplier, **kwargs_copy)


def handle_adx(df, indicator_func, **indicator_kwargs):
    """Handle ADX indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for ADX calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    window = kwargs_copy.pop('window', 14)  # Default ADX window
    
    # Remove HLC string names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)

    # Calculate indicator
    return indicator_func(df['High'], df['Low'], df['Close'], window=window, **kwargs_copy)


def handle_psar(df, indicator_func, **indicator_kwargs):
    """Handle Parabolic SAR indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns for PSAR calculation.")
    
    # Check for Close column (optional for PSAR)
    if 'Close' not in df.columns:
        print("Warning: 'Close' column not found for PSAR. Will use average of High and Low.")
        close = None
    else:
        close = df['Close']
        
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    af_initial = kwargs_copy.pop('af_initial', 0.02)
    af_step = kwargs_copy.pop('af_step', 0.02)
    af_max = kwargs_copy.pop('af_max', 0.2)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], close, af_initial=af_initial, 
                         af_step=af_step, af_max=af_max, **kwargs_copy)


def handle_ichimoku(df, indicator_func, indicator, **indicator_kwargs):
    """Handle Ichimoku Cloud and its components."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for Ichimoku calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    tenkan_period = kwargs_copy.pop('tenkan_period', 9)  # Default tenkan period
    kijun_period = kwargs_copy.pop('kijun_period', 26)  # Default kijun period
    senkou_b_period = kwargs_copy.pop('senkou_b_period', 52)  # Default senkou span B period
    displacement = kwargs_copy.pop('displacement', 26)  # Default displacement
    
    # Remove column names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    kwargs_copy.pop('close', None)
    
    # Calculate full Ichimoku or just the component requested
    if indicator == 'ichimoku':
        return indicator_func(df['High'], df['Low'], df['Close'], 
                          tenkan_period=tenkan_period, 
                          kijun_period=kijun_period, 
                          senkou_b_period=senkou_b_period, 
                          displacement=displacement, 
                          **kwargs_copy)
    else:  # Individual components
        # For individual components, we need all periods except for specific cases
        component_func = indicator_func
        
        if indicator in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']:
            return component_func(df['High'], df['Low'], 
                               tenkan_period=tenkan_period, 
                               kijun_period=kijun_period, 
                               senkou_b_period=senkou_b_period, 
                               displacement=displacement, 
                               **kwargs_copy)
        elif indicator == 'chikou_span':
            return component_func(df['Close'], displacement=displacement, **kwargs_copy)


def handle_aroon(df, indicator_func, **indicator_kwargs):
    """Handle Aroon indicator calculation."""
    # Validate required columns
    if not all(col in df.columns for col in ['High', 'Low']):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns for Aroon calculation.")
    
    # Extract parameters with defaults
    kwargs_copy = indicator_kwargs.copy()
    period = kwargs_copy.pop('period', 14)  # Default period
    
    # Remove column names from kwargs if they were passed
    kwargs_copy.pop('high', None)
    kwargs_copy.pop('low', None)
    
    # Calculate indicator
    return indicator_func(df['High'], df['Low'], period=period, **kwargs_copy)


def format_trend_indicator_name(indicator, kwargs):
    """Format the display name for trend indicators."""
    # Initialize suffix string
    suffix = ""
    
    # Add appropriate parameter values to suffix
    if indicator in ['sma', 'ema', 'wma', 'hma', 'trix']:
        # For moving averages, include the window in the name
        window = kwargs.get('window', 20) # Default for most MAs
        suffix = f"_{window}"
    elif indicator == 'adx':
        # For ADX, include the period
        window = kwargs.get('window', 14)
        suffix = f"_{window}"
    elif indicator == 'aroon':
        # For Aroon, include the period
        period = kwargs.get('period', 14)
        suffix = f"_{period}"
    
    return suffix
