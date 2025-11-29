import pandas as pd


def efr(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Efficiency Ratio (ER), also known as Kaufman Efficiency, which
    measures the efficiency of price movement by comparing net price change to the
    sum of absolute price changes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Efficiency Ratio series and a list of column names.

    The Efficiency Ratio is calculated as follows:

    1. Calculate Net Price Change:
       Change = abs(Close - Close[period ago])

    2. Calculate Volatility (Sum of absolute changes):
       Volatility = Sum(abs(Close - Close[previous])) over period

    3. Calculate Ratio:
       ER = Change / Volatility

    Interpretation:
    - ER near 1.0: Highly efficient, strong trending market.
    - ER near 0.0: Inefficient, choppy/sideways market.
    - ER > 0.7: Strong trend.
    - ER < 0.3: Choppy/Range.

    Use Cases:
    - Trend vs. noise identification: Distinguish trending from ranging markets.
    - Strategy selection: Select trend-following vs mean-reversion systems.
    - Adaptive indicators: Used in KAMA and other adaptive averages.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 10))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate net price change over period
    net_change = (close - close.shift(period)).abs()
    
    # Calculate sum of absolute price changes
    price_changes = close.diff().abs()
    sum_changes = price_changes.rolling(window=period).sum()
    
    # Calculate Efficiency Ratio
    er_values = net_change / sum_changes
    
    # Ensure values are between 0 and 1
    er_values = er_values.clip(0, 1)
    
    er_values.name = f'ER_{period}'
    columns_list = [er_values.name]
    return er_values, columns_list


def strategy_efr(
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
    EFR (Efficiency Ratio) - Trend vs Noise Strategy
    
    LOGIC: Buy when ER rises above upper threshold (trending market),
           sell when drops below lower threshold (choppy market).
    WHY: ER measures efficiency of price movement. High ER indicates
         strong trending, low ER indicates choppy/sideways market.
    BEST MARKETS: All markets. Use to filter trend-following strategies.
    TIMEFRAME: Daily charts. 10-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 10), 'upper' (default 0.7),
                    'lower' (default 0.3)
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
    
    period = int(parameters.get('period', 10))
    upper = float(parameters.get('upper', 0.7))
    lower = float(parameters.get('lower', 0.3))
    price_col = 'Close'
    indicator_col = f'ER_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='efr',
        parameters={"period": period},
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
