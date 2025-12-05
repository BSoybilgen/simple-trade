import numpy as np
import pandas as pd


def tsf(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Time Series Forecast (tsf).
    tsf is a linear regression-based indicator that projects the linear regression
    line one period into the future. It represents the expected price based on
    the current trend.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TSF series and a list of column names.

    The tsf is calculated as follows:

    1. Perform Linear Regression over the rolling window:
       y = mx + c
       where y is price and x is time (0 to window-1).

    2. Calculate the Forecast value for the next period:
       TSF = m * window + c
       (This is one period ahead of the LSM endpoint)

    Interpretation:
    - TSF predicts the next period's price based on linear regression.
    - Price above TSF suggests bullish continuation.
    - Price below TSF suggests bearish continuation.

    Use Cases:
    - Trend Prediction: Forecast where price should be.
    - Reversal Detection: Price diverging from TSF may signal reversal.
    - Support/Resistance: TSF can act as dynamic S/R level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 14))

    series = df[close_col]

    def time_series_forecast(x):
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

        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == 0:
            return np.nan

        # Calculate slope (b) and intercept (a)
        b = (n * sum_tx - sum_t * sum_x) / denominator
        a = (sum_x - b * sum_t) / n

        # Return forecast for the next point (one period ahead)
        return a + b * n

    tsf_series = series.rolling(window=window).apply(time_series_forecast, raw=True)
    tsf_series.name = f'TSF_{window}'

    columns_list = [tsf_series.name]
    return tsf_series, columns_list


def strategy_tsf(
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
    tsf (Time Series Forecast) - Price vs TSF Crossover Strategy
    
    LOGIC: Buy when price crosses above TSF, sell when crosses below.
    WHY: TSF forecasts the next period's price using linear regression.
         Price above TSF indicates bullish momentum, below indicates bearish.
    BEST MARKETS: Trending markets. Stocks, forex, futures. Good for
                  identifying continuation and reversal points.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14)
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
    
    window = int(parameters.get('window', 14))
    price_col = 'Close'
    indicator_col = f'TSF_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='tsf',
        parameters={"window": window},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='Close',
        long_window_indicator=indicator_col,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', indicator_col]
    
    return results, portfolio, indicator_cols_to_plot, data
