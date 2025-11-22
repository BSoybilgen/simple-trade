import pandas as pd
import numpy as np

def psa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Parabolic SAR (PSAR).
    Parabolic SAR (Stop And Reverse) is a trend-following indicator developed by J. Welles Wilder
    that helps identify potential reversals in price direction. It appears as a series of dots
    placed either above or below the price, depending on the trend direction.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - af_initial (float): Initial acceleration factor. Default is 0.02.
            - af_step (float): Acceleration factor step. Default is 0.02.
            - af_max (float): Maximum acceleration factor. Default is 0.2.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with PSAR values and a list of column names.

    Calculation Steps:
    1. Initial SAR value:
       - In an uptrend, SAR starts at the lowest low of the previous data range.
       - In a downtrend, SAR starts at the highest high of the previous data range.

    2. Extreme Points (EP):
       - Uptrend: EP is the highest high reached during the current trend.
       - Downtrend: EP is the lowest low reached during the current trend.

    3. Acceleration Factor (AF):
       - Starts at af_initial. Increases by af_step when a new EP is reached.
       - Capped at af_max.

    4. SAR Calculation:
       SAR = Previous SAR + AF * (EP - Previous SAR)
       (Constraints: SAR cannot be above/below previous period's High/Low depending on trend).

    5. Trend Reversal:
       - When price crosses SAR, trend reverses. SAR resets to EP.

    Interpretation:
    - Dots Below Price: Uptrend (Bullish).
    - Dots Above Price: Downtrend (Bearish).
    - The gap between price and dots tightens as the trend matures (acceleration).

    Use Cases:
    - Stop loss placement: The SAR value can be used as a trailing stop loss.
    - Exit signal generation: A cross of price through the SAR dots indicates a potential reversal.
    - Trend identification: Determining the current market bias.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    af_initial = float(parameters.get('af_initial', 0.02))
    af_step = float(parameters.get('af_step', 0.02))
    af_max = float(parameters.get('af_max', 0.2))
    
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

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
        return result, list(result.columns)

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
    
    columns_list = list(result.columns)
    return result, columns_list
