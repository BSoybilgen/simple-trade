import pandas as pd


def psy(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Psychological Line (psy), a ratio of the number of rising periods 
    over the last N periods to the total number of periods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period N. Default is 12.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PSY series and a list of column names.

    The Psychological Line is calculated as follows:

    1. Identify Rising Periods:
       Rising = 1 if Close > Prev Close else 0

    2. Sum Rising Periods over Window:
       Sum Rising = Sum(Rising, window)

    3. Calculate PSY:
       PSY = (Sum Rising / window) * 100

    Interpretation:
    - Range: 0 to 100.
    - Equilibrium: 50 indicates a balance between rising and falling periods.
    - Overbought: Values above 75 indicate potential overbought conditions.
    - Oversold: Values below 25 indicate potential oversold conditions.

    Use Cases:
    - Sentiment Analysis: Gauging market sentiment (bullish vs bearish days).
    - Reversal Signals: Extreme values suggest that the current trend may be overextended.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 12))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    # Rising periods: Close > Prev Close
    is_rising = (close > close.shift(1)).astype(float)

    psy_val = is_rising.rolling(window=window).sum() / window * 100
    psy_val.name = f'PSY_{window}'

    return psy_val, [psy_val.name]


def strategy_psy(
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
    psy (Psychological Line) - Mean Reversion Strategy
    
    LOGIC: Buy when psy drops below 25 (oversold), sell when rises above 75 (overbought).
    WHY: psy measures ratio of up days to total days. Extreme readings suggest
         sentiment has become one-sided and may revert. Simple but effective.
    BEST MARKETS: Range-bound markets and indices. Good for gauging short-term
                  sentiment extremes. Works well on ETFs and broad market indices.
    TIMEFRAME: Daily charts. 12-period is common. Adjust thresholds based on market.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 12), 'upper' (default 75), 'lower' (default 25)
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
    
    window = int(parameters.get('window', 12))
    upper = int(parameters.get('upper', 75))
    lower = int(parameters.get('lower', 25))
    
    indicator_params = {"window": window}
    indicator_col = f'PSY_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='psy',
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
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
