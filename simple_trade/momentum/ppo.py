import pandas as pd


def ppo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Percentage Price Oscillator (ppo), a momentum indicator similar to MACD.
    It measures the difference between two moving averages as a percentage of the larger moving average.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_window (int): The window size for the fast EMA. Default is 12.
            - slow_window (int): The window size for the slow EMA. Default is 26.
            - signal_window (int): The window size for the signal line EMA. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PPO DataFrame (PPO, Signal, Hist) and a list of column names.

    The Percentage Price Oscillator is calculated as follows:

    1. Calculate Fast and Slow EMAs:
       Fast EMA = EMA(Close, fast_window)
       Slow EMA = EMA(Close, slow_window)

    2. Calculate PPO Line:
       PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100

    3. Calculate Signal Line:
       Signal = EMA(PPO, signal_window)

    4. Calculate Histogram:
       Histogram = PPO - Signal

    Interpretation:
    - Same as MACD but in percentage terms, allowing comparison across securities with different prices.
    - Crossovers: PPO crossing above Signal is bullish; below is bearish.
    - Zero Line: Crossing zero signals a change in trend direction (Short-term average crossing Long-term average).

    Use Cases:
    - Asset Comparison: comparing momentum between assets of different prices.
    - Trend Confirmation: Validating trend direction and strength.
    - Divergence: Identifying potential reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    fast_window_param = parameters.get('fast_window')
    fast_period_param = parameters.get('fast_period')
    if fast_window_param is None and fast_period_param is not None:
        fast_window_param = fast_period_param
    elif fast_window_param is not None and fast_period_param is not None:
        if int(fast_window_param) != int(fast_period_param):
            raise ValueError("Provide either 'fast_window' or 'fast_period' (aliases) with the same value if both are set.")

    slow_window_param = parameters.get('slow_window')
    slow_period_param = parameters.get('slow_period')
    if slow_window_param is None and slow_period_param is not None:
        slow_window_param = slow_period_param
    elif slow_window_param is not None and slow_period_param is not None:
        if int(slow_window_param) != int(slow_period_param):
            raise ValueError("Provide either 'slow_window' or 'slow_period' (aliases) with the same value if both are set.")

    signal_window_param = parameters.get('signal_window')
    signal_period_param = parameters.get('signal_period')
    if signal_window_param is None and signal_period_param is not None:
        signal_window_param = signal_period_param
    elif signal_window_param is not None and signal_period_param is not None:
        if int(signal_window_param) != int(signal_period_param):
            raise ValueError("Provide either 'signal_window' or 'signal_period' (aliases) with the same value if both are set.")

    fast_window = int(fast_window_param if fast_window_param is not None else 12)
    slow_window = int(slow_window_param if slow_window_param is not None else 26)
    signal_window = int(signal_window_param if signal_window_param is not None else 9)
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    ema_fast = close.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=slow_window, adjust=False).mean()

    ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
    signal_line = ppo_line.ewm(span=signal_window, adjust=False).mean()
    histogram = ppo_line - signal_line

    ppo_col = f'PPO_{fast_window}_{slow_window}'
    sig_col = f'PPO_SIG_{signal_window}'
    hist_col = 'PPO_HIST'

    result = pd.DataFrame({
        ppo_col: ppo_line,
        sig_col: signal_line,
        hist_col: histogram
    })

    return result, list(result.columns)


def strategy_ppo(
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
    ppo (Percentage Price Oscillator) - Signal Line Crossover Strategy
    
    LOGIC: Buy when ppo crosses above signal line, sell when crosses below.
    WHY: ppo is MACD in percentage terms, allowing comparison across different
         price levels. Signal crossovers indicate momentum shifts.
    BEST MARKETS: All markets. Particularly useful for comparing momentum across
                  assets with different price levels (e.g., $10 stock vs $1000 stock).
    TIMEFRAME: All timeframes. Standard 12/26/9 settings work well for daily charts.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_window' (default 12), 'slow_window' (default 26),
                   'signal_window' (default 9)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_cross_trade_strategies import run_cross_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}

    fast_window_param = parameters.get('fast_window')
    fast_period_param = parameters.get('fast_period')
    if fast_window_param is None and fast_period_param is not None:
        fast_window_param = fast_period_param
    elif fast_window_param is not None and fast_period_param is not None:
        if int(fast_window_param) != int(fast_period_param):
            raise ValueError("Provide either 'fast_window' or 'fast_period' (aliases) with the same value if both are set.")

    slow_window_param = parameters.get('slow_window')
    slow_period_param = parameters.get('slow_period')
    if slow_window_param is None and slow_period_param is not None:
        slow_window_param = slow_period_param
    elif slow_window_param is not None and slow_period_param is not None:
        if int(slow_window_param) != int(slow_period_param):
            raise ValueError("Provide either 'slow_window' or 'slow_period' (aliases) with the same value if both are set.")

    signal_window_param = parameters.get('signal_window')
    signal_period_param = parameters.get('signal_period')
    if signal_window_param is None and signal_period_param is not None:
        signal_window_param = signal_period_param
    elif signal_window_param is not None and signal_period_param is not None:
        if int(signal_window_param) != int(signal_period_param):
            raise ValueError("Provide either 'signal_window' or 'signal_period' (aliases) with the same value if both are set.")

    fast_window = int(fast_window_param if fast_window_param is not None else 12)
    slow_window = int(slow_window_param if slow_window_param is not None else 26)
    signal_window = int(signal_window_param if signal_window_param is not None else 9)

    indicator_params = {
        "fast_window": fast_window,
        "slow_window": slow_window,
        "signal_window": signal_window
    }
    short_window_indicator = f'PPO_{fast_window}_{slow_window}'
    long_window_indicator = f'PPO_SIG_{signal_window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='ppo',
        parameters=indicator_params,
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
