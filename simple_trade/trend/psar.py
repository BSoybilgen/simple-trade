import pandas as pd
import numpy as np

def psar(df: pd.DataFrame, af_initial=0.02, af_step=0.02, af_max=0.2, high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates Parabolic SAR (PSAR).

    Parabolic SAR (Stop And Reverse) is a trend-following indicator developed by J. Welles Wilder
    that helps identify potential reversals in price direction. It appears as a series of dots
    placed either above or below the price, depending on the trend direction.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high, low, and close columns.
        af_initial (float): Initial acceleration factor. Default is 0.02.
        af_step (float): Acceleration factor step increment. Default is 0.02.
        af_max (float): Maximum acceleration factor. Default is 0.2.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').
        close_col (str): The name of the close price column (default: 'Close').

    Returns:
        pd.DataFrame: DataFrame containing three columns:
            - PSAR: Combined PSAR values
            - PSAR_Bullish: PSAR values during bullish trend (NaN values replaced with 1.5x the price)
            - PSAR_Bearish: PSAR values during bearish trend (NaN values replaced with 0.5x the price)

    The Parabolic SAR is calculated using the following rules:

    1. Initial SAR value:
       - In an uptrend, SAR starts at the lowest low of the data range
       - In a downtrend, SAR starts at the highest high of the data range

    2. Extreme Points (EP):
       - In an uptrend, EP is the highest high reached during the current trend
       - In a downtrend, EP is the lowest low reached during the current trend

    3. Acceleration Factor (AF):
       - Starts at a value specified by af_initial (default 0.02)
       - Increases by af_step (default 0.02) each time a new EP is reached
       - Capped at a maximum of af_max (default 0.2)

    4. SAR Calculation:
       - Current SAR = Previous SAR + AF * (EP - Previous SAR)
       - In an uptrend, SAR cannot be above the low of the previous two periods
       - In a downtrend, SAR cannot be below the high of the previous two periods

    5. Trend Reversal:
       - When price crosses the SAR value, the trend is considered to have reversed
       - SAR then flips to the opposite side of price, and calculations continue with new trend direction

    Use Cases:

    - Trend identification: When dots are below price, the trend is up.
      When dots are above price, the trend is down.

    - Stop loss placement: The SAR value can be used as a trailing stop loss.

    - Exit signal generation: A cross of price through the SAR dots indicates a potential reversal 
      and can be used as a signal to exit the position.

    - Volatility adaptation: Since the acceleration factor increases as the trend develops, 
      the indicator adapts to changes in market volatility.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    length = len(high)
    if length < 2: # Need at least 2 points
        # Return empty DataFrame with all NaN values
        result = pd.DataFrame(
            {
                'PSAR': np.nan,
                'PSAR_Bullish': np.nan,
                'PSAR_Bearish': np.nan
            }, 
            index=high.index
        )
        return result

    psar_values = np.zeros(length)
    psar_bullish = np.full(length, np.nan)  # Initialize with NaN
    psar_bearish = np.full(length, np.nan)  # Initialize with NaN
    trend_is_bull = np.zeros(length, dtype=bool)  # Track the trend direction
    
    bull = True # Initial trend assumption
    af = af_initial
    ep = high.iloc[0] # Initial Extreme Point (assuming initial uptrend)

    # Initialize first SAR value
    psar_values[0] = low.iloc[0]
    trend_is_bull[0] = bull
    
    # Set initial values based on initial trend
    if bull:
        psar_bullish[0] = psar_values[0]
    else:
        psar_bearish[0] = psar_values[0]

    # A slightly more robust initial trend check (optional, needs 'Close' if used)
    # if length > 1 and close.iloc[1] < close.iloc[0]:
    #     bull = False
    #     ep = low.iloc[0]
    #     psar_values[0] = high.iloc[0]

    for i in range(1, length):
        prev_psar = psar_values[i-1]
        prev_ep = ep
        prev_af = af

        if bull:
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            # SAR cannot be higher than the low of the previous two periods
            current_psar = min(current_psar, low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

            if low.iloc[i] < current_psar: # Trend reversal to Bear
                bull = False
                current_psar = prev_ep # SAR starts at the last extreme high
                ep = low.iloc[i]     # New extreme point is the current low
                af = af_initial # Reset AF
            else: # Continue Bull trend
                # If new high is made, update EP and increment AF
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(prev_af + af_step, af_max)
                else:
                    af = prev_af # AF doesn't change if EP not exceeded
        else: # Bear trend
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            # SAR cannot be lower than the high of the previous two periods
            current_psar = max(current_psar, high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

            if high.iloc[i] > current_psar: # Trend reversal to Bull
                bull = True
                current_psar = prev_ep # SAR starts at the last extreme low
                ep = high.iloc[i]     # New extreme point is the current high
                af = af_initial # Reset AF
            else: # Continue Bear trend
                # If new low is made, update EP and increment AF
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(prev_af + af_step, af_max)
                else:
                    af = prev_af # AF doesn't change if EP not exceeded

        psar_values[i] = current_psar
        trend_is_bull[i] = bull
        
        # Populate the appropriate trend-specific array
        if bull:
            psar_bullish[i] = current_psar
        else:
            psar_bearish[i] = current_psar

    # Create a DataFrame with the base PSAR values
    result = pd.DataFrame(
        {
            f'PSAR_{af_initial}_{af_step}_{af_max}': psar_values,
            f'PSAR_Bullish_{af_initial}_{af_step}_{af_max}': psar_bullish,
            f'PSAR_Bearish_{af_initial}_{af_step}_{af_max}': psar_bearish
        }, 
        index=high.index
    )
    
    # Replace NaN values in PSAR_Bullish with half the price and PSAR_Bearish with 1.5x price
    result[f'PSAR_Bullish_{af_initial}_{af_step}_{af_max}'] = result[f'PSAR_Bullish_{af_initial}_{af_step}_{af_max}'].fillna(close * 1.5)
    result[f'PSAR_Bearish_{af_initial}_{af_step}_{af_max}'] = result[f'PSAR_Bearish_{af_initial}_{af_step}_{af_max}'].fillna(close * 0.5)
    
    return result
