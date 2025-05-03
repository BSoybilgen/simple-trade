import pandas as pd
import numpy as np


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator.

    The Supertrend is a trend-following indicator similar to moving averages
    but with better trend identification capability. It plots a line above or below
    the price depending on the current trend direction.

    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        close (pd.Series): The closing prices of the period.
        period (int): The period for ATR calculation. Default is 10.
        multiplier (float): The ATR multiplier. Default is 3.0.

    Returns:
        pd.DataFrame: A DataFrame containing the Supertrend values, upper band,
                      lower band, and trend direction.

    The Supertrend indicator is calculated as follows:

    1. Calculate the Average True Range (ATR) over the specified period.
    2. Calculate the basic upper and lower bands:
       - Basic Upper Band = (High + Low) / 2 + (multiplier * ATR)
       - Basic Lower Band = (High + Low) / 2 - (multiplier * ATR)
    3. Apply a trending rule to finalize the bands:
       - If the current close price crosses above the previous final upper band,
         the current final lower band becomes the basic lower band, and the final
         upper band is set to infinity.
       - If the current close price crosses below the previous final lower band,
         the current final upper band becomes the basic upper band, and the final
         lower band is set to zero.
       - Otherwise, the final bands remain the same as the previous period.
    4. The Supertrend line is either the final upper band (in a downtrend) or
       the final lower band (in an uptrend).

    Use Cases:

    - Trend identification: The Supertrend line helps identify the current market
      trend (uptrend when price is above the line, downtrend when below).
    - Trade signals: Generates buy signals when the price crosses above the
      Supertrend line and sell signals when it crosses below.
    - Stop loss placement: The Supertrend line can be used as a trailing stop loss.
    - Confirmation of other indicators: Can be used in conjunction with other
      indicators to confirm trading signals.
    """
    # Calculating ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculating basic upper and lower bands
    hl_avg = (high + low) / 2
    upper_band_basic = hl_avg + (multiplier * atr)
    lower_band_basic = hl_avg - (multiplier * atr)
    
    # Initialize final bands and trend
    upper_band_final = [0.0] * len(close)
    lower_band_final = [0.0] * len(close)
    supertrend_values = [0.0] * len(close)
    supertrend_bullish = [np.nan] * len(close)  # For uptrend phases
    supertrend_bearish = [np.nan] * len(close)  # For downtrend phases
    trend = [0] * len(close)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend using the trending rule
    for i in range(1, len(close)):
        if isnan := (pd.isna(upper_band_basic.iloc[i]) or pd.isna(lower_band_basic.iloc[i])):
            upper_band_final[i] = np.nan
            lower_band_final[i] = np.nan
            supertrend_values[i] = np.nan
            trend[i] = 0  # No trend when NaN values are present
            continue
            
        # Default: extend the previous trend
        upper_band_final[i] = upper_band_basic.iloc[i]
        lower_band_final[i] = lower_band_basic.iloc[i]
        
        # Previous close crossed above previous upper band
        if close.iloc[i-1] <= upper_band_final[i-1] and close.iloc[i] > upper_band_basic.iloc[i]:
            upper_band_final[i] = upper_band_basic.iloc[i]
            
        # Previous close crossed below previous lower band
        if close.iloc[i-1] >= lower_band_final[i-1] and close.iloc[i] < lower_band_basic.iloc[i]:
            lower_band_final[i] = lower_band_basic.iloc[i]
        
        # Trend continuation criteria
        if upper_band_basic.iloc[i] < upper_band_final[i-1]:
            upper_band_final[i] = upper_band_final[i-1]
        
        if lower_band_basic.iloc[i] > lower_band_final[i-1]:
            lower_band_final[i] = lower_band_final[i-1]
        
        # Determine trend
        if close.iloc[i] > upper_band_final[i-1]:
            trend[i] = 1  # Uptrend
        elif close.iloc[i] < lower_band_final[i-1]:
            trend[i] = -1  # Downtrend
        else:
            trend[i] = trend[i-1]  # Continue previous trend
        
        # Determine Supertrend value based on trend
        if trend[i] == 1:  # Uptrend (bullish)
            supertrend_values[i] = lower_band_final[i]
            supertrend_bullish[i] = lower_band_final[i]  # Store in bullish array
        else:  # Downtrend (bearish)
            supertrend_values[i] = upper_band_final[i]
            supertrend_bearish[i] = upper_band_final[i]  # Store in bearish array
    
    # Create result DataFrame with separate bullish and bearish columns like PSAR
    df_supertrend = pd.DataFrame({
        f'SUPERTREND_{period}_{multiplier}': pd.Series(supertrend_values, index=close.index),
        f'STREND_Bullish_{period}_{multiplier}': pd.Series(supertrend_bullish, index=close.index),
        f'STREND_Bearish_{period}_{multiplier}': pd.Series(supertrend_bearish, index=close.index)
    })
    
    # Fill NaN values in bullish/bearish columns for visualization
    # Similar to PSAR implementation
    df_supertrend[f'STREND_Bullish_{period}_{multiplier}'] = df_supertrend[f'STREND_Bullish_{period}_{multiplier}'].fillna(close * 1.5)
    df_supertrend[f'STREND_Bearish_{period}_{multiplier}'] = df_supertrend[f'STREND_Bearish_{period}_{multiplier}'].fillna(close * 0.5)
    
    return df_supertrend
