import pandas as pd


def sto(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Stochastic Oscillator (sto), a momentum indicator that compares a security's 
    closing price to its price range over a given time period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - k_period (int): The lookback period for %K calculation. Default is 14.
            - d_period (int): The period for %D (the moving average of %K). Default is 3.
            - smooth_k (int): The period for smoothing %K. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Stochastic Oscillator DataFrame (%K and %D) and a list of column names.

    Calculation Steps:
    1. Calculate the raw %K ("Fast Stochastic Oscillator"):
       %K = 100 * ((Current Close - Lowest Low) / (Highest High - Lowest Low))
       where Lowest Low and Highest High are calculated over the last k_period periods.

    2. Calculate the "Full" or "Slow" %K (optional smoothing of raw %K):
       Slow %K = SMA(Fast %K, smooth_k) (if smooth_k > 1)

    3. Calculate %D:
       %D = SMA(Slow %K, d_period)
       %D is essentially a moving average of %K.

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Readings above 80 are considered overbought.
    - Oversold: Readings below 20 are considered oversold.

    Use Cases:
    - Identifying overbought/oversold conditions: Potential reversal zones.
    - Signal line crossovers: When %K crosses above %D (Bullish) or below %D (Bearish).
    - Divergence analysis: If price makes a new high or low but the Stochastic doesn't,
      it may indicate a potential reversal.
    - Identifying trend reversals: Crossovers of 20 and 80 levels.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    k_window_param = parameters.get('k_window')
    k_period_param = parameters.get('k_period')
    if k_window_param is None and k_period_param is not None:
        k_window_param = k_period_param
    elif k_window_param is not None and k_period_param is not None:
        if int(k_window_param) != int(k_period_param):
            raise ValueError("Provide either 'k_window' or 'k_period' (aliases) with the same value if both are set.")

    d_window_param = parameters.get('d_window')
    d_period_param = parameters.get('d_period')
    if d_window_param is None and d_period_param is not None:
        d_window_param = d_period_param
    elif d_window_param is not None and d_period_param is not None:
        if int(d_window_param) != int(d_period_param):
            raise ValueError("Provide either 'd_window' or 'd_period' (aliases) with the same value if both are set.")

    k_period = int(k_window_param if k_window_param is not None else 14)
    d_period = int(d_window_param if d_window_param is not None else 3)
    smooth_k = int(parameters.get('smooth_k', 3))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
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
        f'STO_K_{k_period}_{d_period}_{smooth_k}': k,
        f'STO_D_{k_period}_{d_period}_{smooth_k}': d
    }, index=close.index)
    columns_list = list(result.columns)
    return result, columns_list


def strategy_sto(
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
    sto (Stochastic Oscillator) - Mean Reversion Strategy
    
    LOGIC: Buy when %K drops below 20 (oversold), sell when above 80 (overbought).
    WHY: sto compares close to high-low range. Low values = close near lows (oversold),
         high values = close near highs (overbought). Classic mean reversion indicator.
    BEST MARKETS: Range-bound markets and consolidating assets. Stocks, forex, commodities.
                  Use with trend filter in trending markets.
    TIMEFRAME: All timeframes. 14/3/3 is standard. Adjust for sensitivity.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'k_period' (default 14), 'd_period' (default 3),
                   'smooth_k' (default 3), 'upper' (default 80), 'lower' (default 20)
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
    
    k_window_param = parameters.get('k_window')
    k_period_param = parameters.get('k_period')
    if k_window_param is None and k_period_param is not None:
        k_window_param = k_period_param
    elif k_window_param is not None and k_period_param is not None:
        if int(k_window_param) != int(k_period_param):
            raise ValueError("Provide either 'k_window' or 'k_period' (aliases) with the same value if both are set.")

    d_window_param = parameters.get('d_window')
    d_period_param = parameters.get('d_period')
    if d_window_param is None and d_period_param is not None:
        d_window_param = d_period_param
    elif d_window_param is not None and d_period_param is not None:
        if int(d_window_param) != int(d_period_param):
            raise ValueError("Provide either 'd_window' or 'd_period' (aliases) with the same value if both are set.")

    k_window = int(k_window_param if k_window_param is not None else 14)
    d_window = int(d_window_param if d_window_param is not None else 3)
    smooth_k = int(parameters.get('smooth_k', 3))
    upper = int(parameters.get('upper', 80))
    lower = int(parameters.get('lower', 20))

    k_period = k_window
    d_period = d_window
    
    indicator_params = {
        "k_period": k_period,
        "d_period": d_period,
        "smooth_k": smooth_k
    }
    indicator_col = f'STO_K_{k_period}_{d_period}_{smooth_k}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='sto',
        parameters=indicator_params,
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
    
    indicator_cols_to_plot = [indicator_col, f'STO_D_{k_period}_{d_period}_{smooth_k}', 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
