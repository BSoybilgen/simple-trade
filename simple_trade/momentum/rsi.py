import pandas as pd

def rsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Strength Index (rsi), a momentum indicator used in technical analysis.
    It measures the magnitude of recent price changes to evaluate overbought or oversold conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for the RSI calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RSI series and a list of column names.

    Calculation Steps:
    1. Calculate the difference between consecutive values in the series (Diff).
    2. Separate gains (Diff > 0) and losses (Diff < 0, as positive).
    3. Calculate the Average Gain and Average Loss over the specified window (Smoothed).
    4. Calculate the Relative Strength (RS):
       RS = Average Gain / Average Loss
    5. Calculate the RSI:
       RSI = 100 - (100 / (1 + RS))

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 are often interpreted as overbought.
    - Oversold: Values below 30 are often interpreted as oversold.
    - Centerline: 50 acts as a neutral level.

    Use Cases:
    - Identifying overbought and oversold conditions: Potential reversal zones.
    - Identifying trend direction: RSI > 50 generally indicates uptrend, < 50 downtrend.
    - Generating buy and sell signals: Divergences between the RSI and price (e.g. Price higher high, RSI lower high).
    - Failure Swings: Specific patterns in RSI that signal reversals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f'RSI_{window}'
    columns_list = [rsi.name]
    return rsi, columns_list


def strategy_rsi(
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
    rsi (Relative Strength Index) - Mean Reversion Strategy
    
    LOGIC: Buy when rsi drops below lower threshold (oversold), sell when above upper.
    WHY: rsi measures speed and magnitude of price changes. Extreme readings suggest
         exhaustion and potential reversal. Most popular momentum oscillator.
    BEST MARKETS: Range-bound markets, stocks in consolidation, forex pairs.
                  Use wider thresholds (80/20) in trending markets to avoid early exits.
    TIMEFRAME: All timeframes. 14-period is standard. Shorter periods = more signals.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 80), 'lower' (default 20)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
    upper = int(parameters.get('upper', 80))
    lower = int(parameters.get('lower', 20))
    
    indicator_params = {"window": window}
    indicator_col = f'RSI_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='rsi',
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