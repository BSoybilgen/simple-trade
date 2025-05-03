import pandas as pd
import numpy as np


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, constant: float = 0.015) -> pd.Series:
    """
    Calculates the Commodity Channel Index (CCI), a momentum oscillator used to identify cyclical trends
    and extreme market conditions.

    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        close (pd.Series): The closing prices of the period.
        window (int): The lookback period for the calculation. Default is 20.
        constant (float): The scaling factor used in the CCI formula. Default is 0.015.

    Returns:
        pd.Series: CCI values for the given input series.

    The CCI is calculated in three steps:

    1. Calculate the Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate the Simple Moving Average of the Typical Price (SMA(TP)):
       SMA(TP) = n-period SMA of TP

    3. Calculate the Mean Deviation (MD):
       MD = Mean of absolute deviations of TP from SMA(TP)

    4. Calculate the CCI:
       CCI = (TP - SMA(TP)) / (constant * MD)

    The constant (0.015) is used to normalize the CCI to make it comparable across different securities.
    
    Use Cases:

    - Identifying overbought/oversold conditions: Values above +100 suggest overbought conditions,
      while values below -100 suggest oversold conditions.
    - Detecting trend strength: Values consistently above +100 indicate a strong uptrend, while
      values consistently below -100 indicate a strong downtrend.
    - Identifying potential reversals: Divergence between CCI and price can signal potential reversals.
    - Generating trading signals: Crossing above/below zero line or +/-100 thresholds can generate signals.
    """
    # Make sure all inputs have the same index
    high = high.copy()
    low = low.copy()
    close = close.copy()
    
    # Calculate the Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate the Simple Moving Average of the Typical Price
    sma_tp = typical_price.rolling(window=window).mean()
    
    # Calculate the Mean Deviation
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    # Avoid division by zero
    mean_deviation = mean_deviation.replace(0, np.nan)
    
    # Calculate the CCI
    cci = (typical_price - sma_tp) / (constant * mean_deviation)
    
    return cci
