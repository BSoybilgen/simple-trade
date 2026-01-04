import numpy as np
import pandas as pd


def lsm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Least Squares Moving Average (lsm).
    lsm is the end point of a linear regression line over a rolling window.
    It provides a statistically-based trend line that minimizes the sum of
    squared errors between the line and the actual prices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the LSM series and a list of column names.

    The lsm is calculated as follows:

    1. Perform Linear Regression over the rolling window:
       y = mx + c
       where y is price and x is time (0 to window-1).

    2. Calculate the Forecast value at the end of the window:
       lsm = m * (window - 1) + c

    Interpretation:
    - lsm fits the data better than SMA or EMA but can overshoot in sudden reversals.
    - Ideally suited for identifying the direction of the primary trend.

    Use Cases:
    - Trend Confirmation: If price is above lsm, trend is up.
    - Reversal Signal: Price crossing lsm suggests a potential trend change.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 20)

    series = df[close_col]

    def linear_regression_forecast(x):
        """Calculate the forecast value from linear regression."""
        if len(x) < 2:
            return np.nan
        
        # Create time index
        t = np.arange(len(x))
        
        # Calculate linear regression coefficients
        # y = a + b*t
        n = len(x)
        sum_t = np.sum(t)
        sum_x = np.sum(x)
        sum_tx = np.sum(t * x)
        sum_t2 = np.sum(t * t)
        
        # Calculate slope (b) and intercept (a)
        b = (n * sum_tx - sum_t * sum_x) / (n * sum_t2 - sum_t * sum_t)
        a = (sum_x - b * sum_t) / n
        
        # Return forecast for the next point (end of window)
        return a + b * (n - 1)

    lsma_series = series.rolling(window=window).apply(linear_regression_forecast, raw=True)
    lsma_series.name = f'LSM_{window}'

    columns_list = [lsma_series.name]
    return lsma_series, columns_list


def strategy_lsm(
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
    lsm (Least Squares MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast lsm crosses above slow lsm, sell when crosses below.
    WHY: lsm uses linear regression endpoint, fitting data better than SMA/EMA.
         Statistically-based trend line that minimizes squared errors.
    BEST MARKETS: Trending markets. Stocks, forex, futures. Good for
                  identifying primary trend direction.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 20)
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

    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_window = int(short_window_param if short_window_param is not None else 10)
    long_window = int(long_window_param if long_window_param is not None else 20)
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'LSM_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='lsm',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'LSM_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='lsm',
        parameters={"window": long_window},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
