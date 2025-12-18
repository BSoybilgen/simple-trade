import numpy as np
import pandas as pd


def ama(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Adaptive Moving Average (ama), also known as Kaufman's Adaptive Moving Average (KAMA).
    ama adjusts its smoothing factor based on market noise using an Efficiency Ratio (ER).

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for Efficiency Ratio. Default is 10.
            - fast_period (int): The fast EMA period limit. Default is 2.
            - slow_period (int): The slow EMA period limit. Default is 30.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the AMA series and a list of column names.

    The ama is calculated as follows:

    1. Calculate Efficiency Ratio (ER):
       Change = Abs(Price - Price(n periods ago))
       Volatility = Sum(Abs(Price - Prev Price), n)
       ER = Change / Volatility

    2. Calculate Smoothing Constant (SC):
       Fast SC = 2 / (fast_period + 1)
       Slow SC = 2 / (slow_period + 1)
       Scaled SC = (ER * (Fast SC - Slow SC) + Slow SC)^2

    3. Calculate ama:
       ama = Previous ama + Scaled SC * (Price - Previous ama)

    Interpretation:
    - When market moves directionally (high ER), ama adapts quickly.
    - When market is choppy (low ER), ama flattens out to avoid false signals.

    Use Cases:
    - Trend Following: Identifying the trend with reduced noise in sideways markets.
    - Stop Loss: The flat nature of ama in ranges makes it a good trailing stop level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    er_window = int(window_param if window_param is not None else 10)
    fast_period = int(parameters.get('fast_period', 2))
    slow_period = int(parameters.get('slow_period', 30))

    close = df[close_col]

    # Efficiency Ratio
    direction = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = direction / volatility
    er = er.fillna(0)

    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    smoothing_constant = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    values = close.to_numpy(dtype=float)
    sc_values = smoothing_constant.to_numpy(dtype=float)
    ama_values = np.full_like(values, np.nan)

    valid_idx = np.where(~np.isnan(values))[0]
    if valid_idx.size:
        start = valid_idx[0]
        ama_values[start] = values[start]
        for i in range(start + 1, len(values)):
            previous = ama_values[i - 1]
            price = values[i]
            sc = sc_values[i]
            if np.isnan(price):
                ama_values[i] = previous
                continue
            if np.isnan(sc):
                sc = 0.0
            ama_values[i] = previous + sc * (price - previous)

    ama_series = pd.Series(ama_values, index=close.index,
                           name=f'AMA_{er_window}_{fast_period}_{slow_period}')

    return ama_series, [ama_series.name]


def strategy_ama(
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
    ama (Kaufman Adaptive Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast ama crosses above slow ama, sell when crosses below.
    WHY: ama adapts its speed based on market efficiency ratio. Fast in trends,
         slow in choppy markets. Automatically adjusts to market conditions.
    BEST MARKETS: Works across all market conditions due to adaptive nature.
                  Stocks, forex, futures. Reduces whipsaws in ranging markets.
    TIMEFRAME: Daily or 4-hour charts. One of the best adaptive MAs available.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 30),
                   'fast_period' (default 2), 'slow_period' (default 30)
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

    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    fast_period_param = parameters.get('fast_period')
    fast_window_param = parameters.get('fast_window')
    if fast_period_param is None and fast_window_param is not None:
        fast_period_param = fast_window_param
    elif fast_period_param is not None and fast_window_param is not None:
        if int(fast_period_param) != int(fast_window_param):
            raise ValueError("Provide either 'fast_period' or 'fast_window' (aliases) with the same value if both are set.")

    slow_period_param = parameters.get('slow_period')
    slow_window_param = parameters.get('slow_window')
    if slow_period_param is None and slow_window_param is not None:
        slow_period_param = slow_window_param
    elif slow_period_param is not None and slow_window_param is not None:
        if int(slow_period_param) != int(slow_window_param):
            raise ValueError("Provide either 'slow_period' or 'slow_window' (aliases) with the same value if both are set.")

    short_window = int(short_window_param if short_window_param is not None else 10)
    long_window = int(long_window_param if long_window_param is not None else 30)
    fast_period = int(fast_period_param if fast_period_param is not None else 2)
    slow_period = int(slow_period_param if slow_period_param is not None else 30)
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'AMA_{short_window}_{fast_period}_{slow_period}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='ama',
            parameters={"window": short_window, "fast_period": fast_period, "slow_period": slow_period},
            figure=False
        )
    
    long_window_indicator = f'AMA_{long_window}_{fast_period}_{slow_period}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='ama',
        parameters={"window": long_window, "fast_period": fast_period, "slow_period": slow_period},
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
