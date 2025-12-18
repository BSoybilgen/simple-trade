import pandas as pd


def var(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Variance (var), a statistical measure of dispersion that quantifies
    the spread of data points around the mean. It is the square of standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
            - ddof (int): Delta Degrees of Freedom. Default is 0 (population variance).
                         Use 1 for sample variance.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Variance series and a list of column names.

    Calculation Steps:
    1. Calculate Mean:
       Calculate the simple moving average over the window.
    2. Calculate Variance:
       Sum of squared differences from the mean / N (or N-ddof).

    Interpretation:
    - High Variance: High volatility, prices are spread out.
    - Low Variance: Low volatility, prices are close to the mean.
    - Variance is more sensitive to outliers than MAD.

    Use Cases:
    - Volatility measurement.
    - Risk assessment.
    - Portfolio optimization (Markowitz).
    - Position sizing adjustment.
    - Variance ratio tests.

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
    ddof = int(parameters.get('ddof', 0))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling variance
    var_values = close.rolling(window=window).var(ddof=ddof)
    
    var_values.name = f'VAR_{window}'
    columns_list = [var_values.name]
    return var_values, columns_list


def strategy_var(
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
    var (Variance) - Volatility Threshold Strategy
    
    LOGIC: Buy when var drops below lower percentile (low volatility squeeze),
           sell when rises above upper percentile (high volatility).
    WHY: Variance measures price dispersion. Low variance indicates consolidation,
         high variance indicates volatility expansion.
    BEST MARKETS: All markets. Good for volatility-based strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses rolling percentile bands since variance is in price^2 units.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_pct' (default 80),
                    'lower_pct' (default 20), 'lookback' (default 100)
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
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = f'VAR_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='var',
        parameters={"window": window},
        figure=False
    )
    
    # Calculate rolling percentile bands for variance
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(lower_pct / 100)
    
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
