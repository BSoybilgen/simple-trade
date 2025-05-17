import pandas as pd
import numpy as np

def supertrend(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> pd.DataFrame:
    """
    Calculates the SuperTrend indicator.
    
    SuperTrend is a trend following indicator similar to moving averages.
    It plots on price charts as a line that follows price but stays a certain
    distance from it, reacting to volatility.
    
    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high, low, and close columns.
        parameters (dict): The parameter dictionary that includes period and multiplier for the ATR.
        columns (dict): The column dictionary that includes high, low, and close column names.
    
    Returns:
        pd.DataFrame: DataFrame with 'supertrend' and 'direction' columns.
        
    The SuperTrend indicator combines Average True Range (ATR) with a multiplier
    to create a dynamic support/resistance line that follows the price trend.
    When price crosses above the SuperTrend line, it signals a potential uptrend.
    When price crosses below the SuperTrend line, it signals a potential downtrend.
    
    Use Cases:
    
    - Trend detection: SuperTrend helps identify the current market trend direction.
    - Trade filtering: Use SuperTrend to only take trades in the direction of the trend.
    - Stop loss placement: The SuperTrend line can serve as a trailing stop level.
    - Swing trading: SuperTrend is effective for identifying entry and exit points in swing trading.
    - Support/resistance: The indicator acts as dynamic support in uptrends and resistance in downtrends.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 14)
    multiplier = parameters.get('multiplier', 3.0)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Calculate True Range
    hl = high - low
    hc = np.abs(high - close.shift(1))
    lc = np.abs(low - close.shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1, skipna=False)
    
    # Calculate ATR
    atr = tr.rolling(window=period, min_periods=1).mean() # Use min_periods=1 for ATR
    
    # Calculate basic bands
    mid_price = (high + low) / 2 # Though not directly used in final ST, useful for initial band calc
    up_band = mid_price + (multiplier * atr)
    low_band = mid_price - (multiplier * atr)
    
    # Create result dataframe
    result = pd.DataFrame(index=df.index)
    result['ATR'] = atr # For reference if needed
    result['Basic_up_band'] = up_band # Basic upper band before adjustment
    result['Basic_low_band'] = low_band # Basic lower band before adjustment
    result['Supertrend'] = np.nan
    result['Direction'] = 0  # 1 for uptrend, -1 for downtrend, 0 for undetermined

    # The first 'period-1' ATR values might be less reliable or NaN if min_periods=period.
    # We start calculations from the first valid ATR.
    # With min_periods=1 for ATR, we can start earlier, but Supertrend logic itself needs a previous ST value.
    # Iterative calculation for SuperTrend
    for i in range(len(df)):
        if i < period -1 : # ATR might not be stable enough or is NaN. Or first period elements for rolling
            result.loc[result.index[i], 'Supertrend'] = np.nan # Or some initial value if preferred
            result.loc[result.index[i], 'Direction'] = 0
            continue

        # Initial SuperTrend and direction value (e.g., at index `period-1` or first calculable)
        if i == period -1: # First point where ATR is based on `period` lookback
                           # or first point to set an initial trend
            if close.iloc[i] > result.loc[result.index[i], 'Basic_low_band']:
                result.loc[result.index[i], 'Direction'] = 1
                result.loc[result.index[i], 'Supertrend'] = result.loc[result.index[i], 'Basic_low_band']
            else: # close <= basic_low_band (could also be close < basic_up_band)
                result.loc[result.index[i], 'Direction'] = -1
                result.loc[result.index[i], 'Supertrend'] = result.loc[result.index[i], 'Basic_up_band']
            continue

        # Previous values
        prev_direction = result.loc[result.index[i-1], 'Direction']
        prev_supertrend = result.loc[result.index[i-1], 'Supertrend']
        
        curr_close = close.iloc[i]
        curr_basic_up_band = result.loc[result.index[i], 'Basic_up_band']
        curr_basic_low_band = result.loc[result.index[i], 'Basic_low_band']
        
        curr_direction = prev_direction
        curr_supertrend = np.nan

        if prev_direction == 1: # Previous trend was UP
            curr_supertrend = max(prev_supertrend, curr_basic_low_band) # Ratchet: ST cannot go down in uptrend
            if curr_close < curr_supertrend: # Price crossed below ST line
                curr_direction = -1 # Change trend to DOWN
                curr_supertrend = curr_basic_up_band # New ST is the upper band
        elif prev_direction == -1: # Previous trend was DOWN
            curr_supertrend = min(prev_supertrend, curr_basic_up_band) # Ratchet: ST cannot go up in downtrend
            if curr_close > curr_supertrend: # Price crossed above ST line
                curr_direction = 1 # Change trend to UP
                curr_supertrend = curr_basic_low_band # New ST is the lower band
        else: # Previous direction was 0 (e.g. initial state before period-1)
            # This case should ideally be handled by the i == period-1 block
            # For robustness, re-evaluate based on current price vs bands
            if curr_close > curr_basic_low_band:
                curr_direction = 1
                curr_supertrend = curr_basic_low_band
            else:
                curr_direction = -1
                curr_supertrend = curr_basic_up_band
                
        result.loc[result.index[i], 'Direction'] = curr_direction
        result.loc[result.index[i], 'Supertrend'] = curr_supertrend

        df = result[['Supertrend', 'Direction']].copy()
        df.rename(columns={'Supertrend': f'Supertrend_{period}_{multiplier}', 
                           'Direction': f'Direction_{period}_{multiplier}'}, 
                           inplace=True)
        
    return df