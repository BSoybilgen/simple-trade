import pandas as pd


def awo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Awesome Oscillator (AO), a momentum indicator used to measure market momentum.
    It calculates the difference between a 34-period and 5-period Simple Moving Average 
    applied to the median price (High+Low)/2.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_window (int): The period for the fast SMA. Default is 5.
            - slow_window (int): The period for the slow SMA. Default is 34.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Awesome Oscillator series and a list of column names.

    The Awesome Oscillator is calculated in the following steps:

    1. Calculate the Median Price:
       Median Price = (High + Low) / 2

    2. Calculate the Fast Simple Moving Average (SMA):
       SMA Fast = SMA(Median Price, fast_window)

    3. Calculate the Slow Simple Moving Average (SMA):
       SMA Slow = SMA(Median Price, slow_window)

    4. Calculate the Awesome Oscillator:
       AO = SMA Fast - SMA Slow

    Interpretation:
    - Positive AO: Fast momentum is greater than slow momentum (Bullish).
    - Negative AO: Fast momentum is less than slow momentum (Bearish).
    - Zero Line Crossover: Crossing above 0 is a buy signal, crossing below 0 is a sell signal.
    - Color coding (often used): Green if AO > Previous AO, Red if AO < Previous AO.

    Use Cases:
    - Momentum Confirmation: Used to confirm the strength of a trend.
    - Signal Generation: Zero line crossovers, "Saucer" signals (three histograms), and "Twin Peaks" (divergence).
    - Divergence: Divergence between price and AO can signal potential reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    fast_window = int(parameters.get('fast_window', 5))
    slow_window = int(parameters.get('slow_window', 34))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    mid_price = (df[high_col] + df[low_col]) / 2

    sma_fast = mid_price.rolling(window=fast_window).mean()
    sma_slow = mid_price.rolling(window=slow_window).mean()

    ao_series = sma_fast - sma_slow
    ao_series.name = f'AO_{fast_window}_{slow_window}'

    return ao_series, [ao_series.name]


def strategy_awo(
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
    AWO (Awesome Oscillator) - Zero Line Crossover Strategy
    
    LOGIC: Buy when AO crosses above zero (bullish momentum), sell when crosses below zero.
    WHY: AO measures market momentum by comparing recent price action to historical.
         Positive AO = fast momentum > slow momentum (bullish), negative = bearish.
    BEST MARKETS: Trending markets with clear directional moves. Works well on stocks,
                  forex, and commodities. Less effective in choppy/ranging markets.
    TIMEFRAME: Daily or weekly charts preferred. Intraday can generate false signals.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_window' (default 5), 'slow_window' (default 34)
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
    
    fast_window = int(parameters.get('fast_window', 5))
    slow_window = int(parameters.get('slow_window', 34))
    
    indicator_params = {"fast_window": fast_window, "slow_window": slow_window}
    short_window_indicator = f'AO_{fast_window}_{slow_window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='awo',
        parameters=indicator_params,
        figure=False
    )
    
    # Create zero line for crossover strategy
    data['zero_line'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, 'zero_line']
    
    return results, portfolio, indicator_cols_to_plot, data
