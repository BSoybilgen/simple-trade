import pandas as pd

def wil(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Williams %R (wil), a momentum oscillator that measures
    overbought and oversold levels based on recent closing prices relative to the high-low range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period used for the highest high and lowest low. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Williams %R series and a list of column names.

    The Williams %R is calculated as follows:

    1. Identify Highest High and Lowest Low over the window.

    2. Calculate %R:
       %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

    Interpretation:
    - Range: 0 to -100.
    - Overbought: Values above -20 (closer to 0) indicate overbought conditions.
    - Oversold: Values below -80 (closer to -100) indicate oversold conditions.
    - Momentum Failure: If price makes a new high but %R fails to move above -20, it signals weak momentum.

    Use Cases:
    - Overbought/Oversold: Identifying potential reversal points.
    - Momentum Confirmation: Strong trends tend to keep %R near the extremes (-20 in uptrends, -80 in downtrends).
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    range_values = (highest_high - lowest_low).where(lambda x: x != 0)

    williams_r = ((highest_high - close) / range_values) * -100
    williams_r.name = f'WIL_{window}'

    columns_list = [williams_r.name]
    return williams_r, columns_list


def strategy_wil(
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
    wil (Williams %R) - Mean Reversion Strategy
    
    LOGIC: Buy when %R drops below -80 (oversold), sell when above -20 (overbought).
    WHY: wil measures close relative to high-low range. Near 0 = overbought
         (close near highs), near -100 = oversold (close near lows).
    BEST MARKETS: Range-bound markets. Stocks, forex, commodities. Similar to
                  Stochastic but inverted scale. Good for mean reversion.
    TIMEFRAME: All timeframes. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default -20), 'lower' (default -80)
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
    
    window = int(parameters.get('window', 14))
    upper = float(parameters.get('upper', -20))
    lower = float(parameters.get('lower', -80))
    
    indicator_params = {"window": window}
    indicator_col = f'WIL_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='wil',
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
