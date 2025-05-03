import pandas as pd
import numpy as np


def trix(series: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Calculates the TRIX (Triple Exponential Average) indicator.

    TRIX is a momentum oscillator that displays the percent rate of change of a triple
    exponentially smoothed moving average. It oscillates around a zero line and can be
    used to identify overbought/oversold conditions, divergences, and trend direction.

    Args:
        series (pd.Series): The input price series, typically close prices.
        window (int): The window size for the EMA calculations. Default is 14.

    Returns:
        pd.DataFrame: DataFrame containing the TRIX line and its signal line.

    The TRIX is calculated through the following steps:
    1. Calculate a single EMA of the price series with the specified window.
    2. Calculate an EMA of the result from step 1 (double-smoothed EMA).
    3. Calculate an EMA of the result from step 2 (triple-smoothed EMA).
    4. Calculate the 1-period percent rate of change of the triple-smoothed EMA.

    The formula can be expressed as:
    TRIX = 100 * (EMA3 - Previous EMA3) / Previous EMA3
    where EMA3 is the triple-smoothed EMA.

    Use Cases:

    - Trend identification: TRIX crossing above zero indicates a bullish trend,
      while crossing below zero indicates a bearish trend.
    - Signal line crossovers: When TRIX crosses above its signal line, it generates
      a bullish signal; when it crosses below, it generates a bearish signal.
    - Divergences: Divergence between TRIX and price can indicate potential
      trend reversals.
    - Filter: TRIX can be used to filter out market noise and identify
      significant market moves.
    """
    # Step 1: Calculate the single-smoothed EMA
    ema1 = series.ewm(span=window, adjust=False).mean()
    
    # Step 2: Calculate the double-smoothed EMA
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    
    # Step 3: Calculate the triple-smoothed EMA
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    # Step 4: Calculate the 1-period percent rate of change of the triple-smoothed EMA
    trix_line = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    
    # Calculate signal line (9-period EMA of TRIX)
    signal_line = trix_line.ewm(span=9, adjust=False).mean()
    
    # Create result DataFrame
    df_trix = pd.DataFrame({
        f'TRIX_{window}': trix_line,
        f'TRIX_SIGNAL_{window}': signal_line
    })
    
    return df_trix
