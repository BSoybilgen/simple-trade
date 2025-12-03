import pandas as pd


def qua(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Rolling Quantile (qua), a statistical measure that divides a distribution
    into intervals with equal probabilities.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
            - quantile (float): The quantile to calculate (0-1). Default is 0.5 (median).
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Quantile series and a list of column names.

    Calculation Steps:
    1. Sort values in the rolling window.
    2. Calculate the value at the specified quantile position.

    Interpretation:
    - Quantile 0.25 (Q1): 25% of values are below this level.
    - Quantile 0.50 (Q2): Median, 50% of values below.
    - Quantile 0.75 (Q3): 75% of values are below this level.
    - Price relative to quantiles indicates distribution position.

    Use Cases:
    - Dynamic support/resistance levels.
    - Percentile-based bands.
    - Distribution analysis.
    - Extreme value detection.
    - Risk management thresholds.

    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    quantile = float(parameters.get('quantile', 0.5))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling quantile
    qua_values = close.rolling(window=window).quantile(quantile)
    
    quantile_pct = int(quantile * 100)
    qua_values.name = f'QUA_{window}_{quantile_pct}'
    columns_list = [qua_values.name]
    return qua_values, columns_list


def strategy_qua(
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
    qua (Rolling Quantile) - Percentile Band Strategy
    
    LOGIC: Buy when price drops below lower qua (oversold),
           sell when price rises above upper qua (overbought).
    WHY: Quantiles provide dynamic support/resistance based on recent distribution.
         Extreme quantiles indicate potential reversal points.
    BEST MARKETS: Range-bound markets. Good for mean reversion strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses upper (75%) and lower (25%) quantiles as bands.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_quantile' (default 0.75),
                    'lower_quantile' (default 0.25)
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
    
    window = int(parameters.get('window', 20))
    upper_quantile = float(parameters.get('upper_quantile', 0.75))
    lower_quantile = float(parameters.get('lower_quantile', 0.25))
    price_col = 'Close'
    
    # Calculate upper quantile
    data, _, _ = compute_indicator(
        data=data,
        indicator='qua',
        parameters={"window": window, "quantile": upper_quantile},
        figure=False
    )
    upper_pct = int(upper_quantile * 100)
    data['upper'] = data[f'QUA_{window}_{upper_pct}']
    
    # Calculate lower quantile
    data, _, _ = compute_indicator(
        data=data,
        indicator='qua',
        parameters={"window": window, "quantile": lower_quantile},
        figure=False
    )
    lower_pct = int(lower_quantile * 100)
    data['lower'] = data[f'QUA_{window}_{lower_pct}']
    
    # Calculate median for plotting
    data, _, _ = compute_indicator(
        data=data,
        indicator='qua',
        parameters={"window": window, "quantile": 0.5},
        figure=False
    )
    indicator_col = f'QUA_{window}_50'
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=price_col,
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
