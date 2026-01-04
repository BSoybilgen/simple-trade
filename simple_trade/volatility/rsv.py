import pandas as pd
import numpy as np


def rsv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Rogers-Satchell Volatility (rsv), which accounts for drift in price movements
    and handles trending markets better than Garman-Klass. Uses all OHLC components.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - trading_periods (int): Trading periods per year. Default is 252.
            - annualized (bool): Whether to annualize. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): Open prices column. Default is 'Open'.
            - high_col (str): High prices column. Default is 'High'.
            - low_col (str): Low prices column. Default is 'Low'.
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: Rogers-Satchell Volatility series and column names list.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hc = ln(High / Close)
       log_ho = ln(High / Open)
       log_lc = ln(Low / Close)
       log_lo = ln(Low / Open)

    2. Calculate RS Component:
       RS = log_hc * log_ho + log_lc * log_lo

    3. Average and Square Root:
       Vol = sqrt(Average(RS, period))

    4. Annualize (optional):
       RSV = Vol * sqrt(trading_periods) * 100

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - Better for trending markets as it allows for non-zero drift.

    Use Cases:
    - Volatility estimation in trending markets.
    - Options pricing and risk management.
    - Advanced volatility analysis.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    open_price = df[open_col]
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Rogers-Satchell formula
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_price)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_price)
    
    rs_component = log_hc * log_ho + log_lc * log_lo
    
    # Rolling mean and square root
    rs_variance = rs_component.rolling(window=period).mean()
    rs_volatility = np.sqrt(rs_variance.abs())
    
    if annualized:
        rs_volatility = rs_volatility * np.sqrt(trading_periods)
    
    rsv_values = rs_volatility * 100
    
    rsv_values.name = f'RSV_{period}_Ann' if annualized else f'RSV_{period}'
    return rsv_values, [rsv_values.name]


def strategy_rsv(
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
    rsv (Rogers-Satchell Volatility) - Volatility Threshold Strategy
    
    LOGIC: Buy when rsv drops below lower threshold (low volatility),
           sell when rises above upper threshold (high volatility).
    WHY: rsv accounts for drift in price movements. Better for trending
         markets than Garman-Klass.
    BEST MARKETS: Trending markets. Good for advanced volatility analysis.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'upper' (default 30),
                    'lower' (default 15)
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
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    upper = float(parameters.get('upper', 30))
    lower = float(parameters.get('lower', 15))
    price_col = 'Close'
    indicator_col = f'RSV_{period}_Ann'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='rsv',
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
