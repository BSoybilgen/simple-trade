import pandas as pd


def svi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Stochastic Volatility Indicator (SVI), which applies the stochastic oscillator
    formula to a volatility measure (typically ATR or standard deviation) to create a normalized
    volatility indicator that oscillates between 0 and 100.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - atr_period (int): The period for ATR calculation. Default is 14.
            - stoch_period (int): The lookback period for stochastic calculation. Default is 14.
            - smooth_k (int): The smoothing period for %K. Default is 3.
            - smooth_d (int): The smoothing period for %D signal line. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with SVI_K and SVI_D columns, and a list of column names.

    Calculation Steps:
    1. Calculate Average True Range (ATR) over atr_period.
    2. Calculate %K (Stochastic of ATR):
       %K = 100 * (Current ATR - Lowest ATR) / (Highest ATR - Lowest ATR)
       (Highest and Lowest over stoch_period).
    3. Smooth %K to get SVI_K:
       SVI_K = SMA(%K, smooth_k).
    4. Calculate Signal Line (SVI_D):
       SVI_D = SMA(SVI_K, smooth_d).

    Interpretation:
    - High SVI (>80): High volatility regime.
    - Low SVI (<20): Low volatility regime.
    - Rising SVI: Increasing volatility.
    - Falling SVI: Decreasing volatility.

    Use Cases:
    - Volatility regime identification.
    - Breakout prediction (from low SVI).
    - Risk management (adjusting size based on regime).
    - Divergence detection.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    atr_period = int(parameters.get('atr_period', 14))
    stoch_period = int(parameters.get('stoch_period', 14))
    smooth_k = int(parameters.get('smooth_k', 3))
    smooth_d = int(parameters.get('smooth_d', 3))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using Wilder's smoothing method
    atr_values = pd.Series(index=close.index, dtype=float)
    first_atr = tr.iloc[:atr_period].mean()
    atr_values.iloc[atr_period-1] = first_atr
    
    for i in range(atr_period, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (atr_period-1)) + tr.iloc[i]) / atr_period
    
    # Apply Stochastic formula to ATR
    lowest_atr = atr_values.rolling(window=stoch_period).min()
    highest_atr = atr_values.rolling(window=stoch_period).max()
    
    # Calculate raw %K
    stoch_k_raw = 100 * (atr_values - lowest_atr) / (highest_atr - lowest_atr)
    
    # Smooth %K to get SVI_K
    svi_k = stoch_k_raw.rolling(window=smooth_k).mean()
    
    # Calculate SVI_D (signal line)
    svi_d = svi_k.rolling(window=smooth_d).mean()
    
    # Create result DataFrame
    result = pd.DataFrame(index=close.index)
    svi_k_name = f'SVI_K_{atr_period}_{stoch_period}_{smooth_k}'
    svi_d_name = f'SVI_D_{atr_period}_{stoch_period}_{smooth_d}'
    
    result[svi_k_name] = svi_k
    result[svi_d_name] = svi_d
    
    columns_list = [svi_k_name, svi_d_name]
    return result, columns_list
