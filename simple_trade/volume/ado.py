import numpy as np


def ado(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Accumulation/Distribution Oscillator (ADO), which measures the momentum
    of the Accumulation/Distribution Line. It helps identify the strength of accumulation
    or distribution by comparing the current A/D value to a past value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for rate of change. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the A/D Oscillator series and a list of column names.

    The Accumulation/Distribution Oscillator is calculated as follows:

    1. Calculate A/D Line:
       See Accumulation/Distribution Line (ADL) calculation.

    2. Calculate Oscillator (Rate of Change):
       ADO = A/D Line - A/D Line(n periods ago)

    Interpretation:
    - Positive ADO: Accumulation pressure is increasing.
    - Negative ADO: Distribution pressure is increasing.
    - Crossing Zero: Signal of change in pressure direction (Buying <-> Selling).

    Use Cases:
    - Momentum Analysis: Gauge the acceleration of money flow.
    - Reversal Detection: Look for divergences between Price and ADO.
    - Trend Strength: Increasing ADO confirms uptrend strength.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = parameters.get('period', 14)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate A/D Line
    price_range = high - low
    price_range_nonzero = price_range.replace(0, np.nan)
    mfm = ((2 * close - high - low) / price_range_nonzero).fillna(0)
    mfv = mfm * volume
    ad_line = mfv.cumsum()
    
    # Calculate A/D Oscillator as rate of change
    ado_values = ad_line - ad_line.shift(period)
    
    ado_values.name = f'ADO_{period}'
    columns_list = [ado_values.name]
    return ado_values, columns_list


def strategy_ado(
    data,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    ADO (Accumulation/Distribution Oscillator) - Zero Line Cross Strategy
    
    LOGIC: Buy when ADO crosses above zero (accumulation momentum),
           sell when ADO crosses below zero (distribution momentum).
    WHY: ADO measures the rate of change of the A/D Line. Positive values indicate
         increasing accumulation, negative values indicate increasing distribution.
    BEST MARKETS: Stocks, ETFs. Good for momentum-based volume analysis.
    TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14)
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
    
    period = int(parameters.get('period', 14))
    price_col = 'Close'
    indicator_col = f'ADO_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='ado',
        parameters={"period": period},
        figure=False
    )
    
    # Create zero line for crossover
    data['zero'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'zero']
    
    return results, portfolio, indicator_cols_to_plot, data
