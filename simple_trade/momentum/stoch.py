import pandas as pd
import numpy as np


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, 
         d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """
    Calculates the Stochastic Oscillator, a momentum indicator that compares a security's 
    closing price to its price range over a given time period.

    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        close (pd.Series): The closing prices of the period.
        k_period (int): The lookback period for %K calculation. Default is 14.
        d_period (int): The period for %D (the moving average of %K). Default is 3.
        smooth_k (int): The period for smoothing %K. Default is 3.

    Returns:
        pd.DataFrame: A DataFrame containing %K and %D values.

    The Stochastic Oscillator is calculated in three steps:

    1. Calculate the raw %K ("Fast Stochastic Oscillator"):
       %K = 100 * ((Current Close - Lowest Low) / (Highest High - Lowest Low))
       where Lowest Low and Highest High are calculated over the last k_period periods.

    2. Calculate the "Full" or "Slow" %K (optional smoothing of raw %K):
       Slow %K = n-period SMA of Fast %K (n is smooth_k)

    3. Calculate %D:
       %D = n-period SMA of %K (n is d_period)
       %D is essentially a moving average of %K.

    The Stochastic oscillates between 0 and 100:
    - Readings above 80 are considered overbought
    - Readings below 20 are considered oversold

    Use Cases:

    - Identifying overbought/oversold conditions: Values above 80 suggest overbought,
      while values below 20 suggest oversold.
    - Identifying trend reversals: When the oscillator crosses above 20, it may signal
      a bullish reversal; when it crosses below 80, it may signal a bearish reversal.
    - Signal line crossovers: When %K crosses above %D, it's often interpreted as a buy
      signal; when %K crosses below %D, it's often interpreted as a sell signal.
    - Divergence analysis: If price makes a new high or low but the Stochastic doesn't,
      it may indicate a potential reversal.
    """
    # Make sure all inputs have the same index
    high = high.copy()
    low = low.copy()
    close = close.copy()
    
    # Find the lowest low and highest high over the lookback period
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Calculate the raw (fast) %K
    fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Apply smoothing to get the "slow" or "full" %K
    k = fast_k.rolling(window=smooth_k).mean() if smooth_k > 1 else fast_k
    
    # Calculate %D (the moving average of %K)
    d = k.rolling(window=d_period).mean()
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        'STOCH_K': k,
        'STOCH_D': d
    }, index=close.index)
    
    return result
