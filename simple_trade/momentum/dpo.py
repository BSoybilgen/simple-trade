import pandas as pd


def dpo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Detrended Price Oscillator (DPO), an indicator designed to remove trend 
    from price and make it easier to identify cycles and overbought/oversold levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the SMA. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DPO series and a list of column names.

    The Detrended Price Oscillator is calculated as follows:

    1. Calculate the Simple Moving Average (SMA):
       SMA = SMA(Close, window)

    2. Calculate the Displacement:
       Displacement = (window / 2) + 1

    3. Calculate DPO:
       DPO = Close - SMA(shifted by Displacement)
       (Note: The formula effectively compares the current price to a past SMA value).

    Interpretation:
    - Positive DPO: Price is above the displaced moving average (Bullish/Overbought).
    - Negative DPO: Price is below the displaced moving average (Bearish/Oversold).
    - Zero Line Crossings: Can signal a change in the short-term trend or cycle.

    Use Cases:
    - Cycle Identification: Isolate short-term cycles by removing long-term trends.
    - Divergence: Identify potential reversals when price and DPO diverge.
    - Overbought/Oversold: Identify extremes within the cycle.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    sma = series.rolling(window=window, min_periods=window).mean()

    displacement = (window // 2) + 1
    displaced_sma = sma.shift(displacement)

    dpo_values = series - displaced_sma
    dpo_values.name = f'DPO_{window}'

    columns_list = [dpo_values.name]
    return dpo_values, columns_list


def strategy_dpo(
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
    DPO (Detrended Price Oscillator) - Zero Line Crossover Strategy
    
    LOGIC: Buy when DPO crosses above zero (price above displaced MA), sell when crosses below.
    WHY: DPO removes trend to isolate cycles. Positive DPO = price above its historical average,
         negative = below. Zero crossings signal cycle turning points.
    BEST MARKETS: Cyclical markets and assets with regular oscillations. Stocks, commodities,
                  and indices with identifiable cycles. Less effective in strong trending markets.
    TIMEFRAME: Daily charts. 20-period is standard. Useful for identifying cycle peaks/troughs.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20)
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
    
    window = int(parameters.get('window', 20))
    
    indicator_params = {"window": window}
    short_window_indicator = f'DPO_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='dpo',
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
