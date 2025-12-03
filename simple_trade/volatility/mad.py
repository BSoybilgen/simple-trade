import pandas as pd


def mad(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Median Absolute Deviation (mad), a robust volatility measure that
    is less sensitive to outliers than standard deviation. It measures dispersion
    around the median rather than the mean.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - scale_factor (float): Scaling factor for normal distribution. Default is 1.4826.

        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: MAD series and column names list.

    Calculation Steps:
    1. Calculate Median:
       Find the median of returns over the rolling window.

    2. Calculate Absolute Deviations:
       AbsDev = |Return - Median|

    3. Calculate MAD:
       MAD = Median(AbsDev) * scale_factor

    Interpretation:
    - Low MAD: Low volatility, stable prices.
    - High MAD: High volatility, large price swings.
    - More robust to outliers than standard deviation.

    Use Cases:
    - Robust volatility measurement for non-normal distributions.
    - Outlier-resistant risk metrics.
    - Better for data with extreme values.

    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    scale_factor = float(parameters.get('scale_factor', 1.4826))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate returns
    returns = close.pct_change()
    
    def calculate_mad(window):
        if len(window) < 2:
            return 0
        median_return = window.median()
        abs_deviations = (window - median_return).abs()
        mad_value = abs_deviations.median() * scale_factor
        return mad_value
    
    mad_values = returns.rolling(window=period).apply(calculate_mad, raw=False)
    
    # Convert to percentage
    mad_values = mad_values * 100
    
    mad_values.name = f'MAD_{period}'
    return mad_values, [mad_values.name]


def strategy_mad(
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
    mad (Median Absolute Deviation) - Volatility Threshold Strategy
    
    LOGIC: Buy when mad drops below lower threshold (low volatility),
           sell when rises above upper threshold (high volatility).
    WHY: mad is more robust to outliers than standard deviation.
         Better for non-normal distributions and extreme values.
    BEST MARKETS: All markets. Good for robust volatility measurement.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'upper' (default 2.0),
                    'lower' (default 0.5)
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
    
    period = int(parameters.get('period', 20))
    upper = float(parameters.get('upper', 2.0))
    lower = float(parameters.get('lower', 0.5))
    price_col = 'Close'
    indicator_col = f'MAD_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='mad',
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
