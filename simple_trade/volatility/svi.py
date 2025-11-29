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


def strategy_svi(
    data: pd.DataFrame,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    SVI (Stochastic Volatility Indicator) - Volatility Regime Strategy
    
    LOGIC: Buy when SVI drops below lower threshold (low volatility regime),
           sell when rises above upper threshold (high volatility regime).
    WHY: SVI applies stochastic formula to ATR. Low SVI indicates low volatility
         regime (potential breakout setup), high SVI indicates high volatility.
    BEST MARKETS: All markets. Good for volatility regime identification.
    TIMEFRAME: Daily charts. 14-period ATR with 14-period stochastic is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'atr_period', 'stoch_period', 'smooth_k', 'smooth_d',
                    'upper' (default 80), 'lower' (default 20)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    atr_period = int(parameters.get('atr_period', 14))
    stoch_period = int(parameters.get('stoch_period', 14))
    smooth_k = int(parameters.get('smooth_k', 3))
    smooth_d = int(parameters.get('smooth_d', 3))
    upper = float(parameters.get('upper', 80))
    lower = float(parameters.get('lower', 20))
    price_col = 'Close'
    indicator_col = f'SVI_K_{atr_period}_{stoch_period}_{smooth_k}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='svi',
        parameters={"atr_period": atr_period, "stoch_period": stoch_period, 
                    "smooth_k": smooth_k, "smooth_d": smooth_d},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
