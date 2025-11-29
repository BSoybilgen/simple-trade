import numpy as np
import pandas as pd


def mgd(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the McGinley Dynamic (MGD) indicator.
    The McGinley Dynamic is a moving average that automatically adjusts its speed
    based on market volatility. It speeds up in fast markets and slows down in
    ranging markets, minimizing whipsaws and price separation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the McGinley Dynamic series and a list of column names.

    The McGinley Dynamic is calculated as follows:

    1. Calculate McGinley Dynamic (MD):
       MD[i] = MD[i-1] + (Price[i] - MD[i-1]) / (N * (Price[i] / MD[i-1])^4)
       where N is the period parameter (window).

    Interpretation:
    - It looks like a moving average but tracks the price much better.
    - It avoids the lag of SMA and the whipsaws of EMA.

    Use Cases:
    - Trend Following: A reliable trend line that hugs the price action.
    - Dynamic Support/Resistance: Often acts as a strong support/resistance level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col].copy()
    
    # Initialize McGinley Dynamic with SMA
    md = pd.Series(index=series.index, dtype=float)
    md.iloc[:window] = series.iloc[:window].rolling(window=window).mean()
    
    # Calculate McGinley Dynamic
    for i in range(window, len(series)):
        if pd.notna(md.iloc[i-1]) and md.iloc[i-1] != 0:
            ratio = series.iloc[i] / md.iloc[i-1]
            # Prevent extreme values
            ratio = np.clip(ratio, 0.1, 10.0)
            k = (series.iloc[i] - md.iloc[i-1]) / (window * (ratio ** 4))
            md.iloc[i] = md.iloc[i-1] + k
        else:
            md.iloc[i] = series.iloc[i]
    
    md.name = f'MGD_{window}'
    
    columns_list = [md.name]
    return md, columns_list


def strategy_mgd(
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
    MGD (McGinley Dynamic) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast MGD crosses above slow MGD, sell when crosses below.
    WHY: MGD automatically adjusts speed based on volatility. Speeds up in
         fast markets, slows in ranging markets. Minimizes whipsaws.
    BEST MARKETS: All market conditions. Stocks, forex, futures.
                  Excellent for avoiding whipsaws.
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
    
    short_window = int(parameters.get('short_window', 10))
    long_window = int(parameters.get('long_window', 20))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'MGD_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='mgd',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'MGD_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='mgd',
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
