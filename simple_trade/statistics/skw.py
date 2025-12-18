import pandas as pd


def skw(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Skewness (skw), a statistical measure of the asymmetry of a distribution
    around its mean. It indicates whether data points are skewed to the left or right.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Skewness series and a list of column names.

    Calculation Steps:
    1. Calculate rolling skewness over the specified window.
    2. Skewness measures the third standardized moment.

    Interpretation:
    - Skewness > 0 (Positive): Right-skewed, tail extends to the right, more extreme gains.
    - Skewness = 0: Symmetric distribution around the mean.
    - Skewness < 0 (Negative): Left-skewed, tail extends to the left, more extreme losses.
    - High absolute skewness indicates asymmetric risk.

    Use Cases:
    - Risk assessment (asymmetric return distribution).
    - Market sentiment analysis.
    - Options pricing adjustments.
    - Portfolio construction.
    - Identifying potential trend exhaustion.

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

    window = int(window_param if window_param is not None else 20)
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling skewness
    skw_values = close.rolling(window=window).skew()
    
    skw_values.name = f'SKW_{window}'
    columns_list = [skw_values.name]
    return skw_values, columns_list


def strategy_skw(
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
    skw (Skewness) - Distribution Asymmetry Strategy
    
    LOGIC: Buy when skw is positive (right-skewed, potential for gains),
           sell when skw is negative (left-skewed, potential for losses).
    WHY: Positive skewness indicates more potential for extreme gains,
         negative skewness indicates more potential for extreme losses.
    BEST MARKETS: All markets. Useful for risk-adjusted strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses zero as the threshold (symmetric distribution).
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_threshold' (default 0.5),
                    'lower_threshold' (default -0.5)
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

    window = int(window_param if window_param is not None else 20)
    upper_threshold = float(parameters.get('upper_threshold', 0.5))
    lower_threshold = float(parameters.get('lower_threshold', -0.5))
    price_col = 'Close'
    indicator_col = f'SKW_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='skw',
        parameters={"window": window},
        figure=False
    )
    
    # Use fixed thresholds for skewness
    data['upper'] = upper_threshold
    data['lower'] = lower_threshold
    
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
