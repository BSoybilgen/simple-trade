import pandas as pd
import numpy as np


def fdi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Fractal Dimension Index (fdi), which measures market complexity
    and choppiness based on fractal geometry. Values near 1.5 indicate random walk.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: FDI series and column names list.

    The FDI is calculated as follows:

    1. Calculate Path Length:
       Sum of absolute differences between consecutive prices over the window.

    2. Calculate Direct Distance:
       Absolute difference between first and last price in the window.

    3. Calculate Fractal Dimension:
       FDI = 1 + (log(Path Length) - log(Direct Distance)) / log(window)

    Interpretation:
    - FDI near 1.0: Highly persistent, trending (Linear).
    - FDI near 1.5: Random walk, no clear pattern (Brownian motion).
    - FDI near 2.0: Anti-persistent, mean-reverting (Jagged).

    Use Cases:
    - Market complexity measurement.
    - Trend vs. random identification.
    - Strategy selection based on market structure.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    def calculate_fdi(window):
        if len(window) < 2:
            return 1.5
        
        n = len(window)
        prices = window.values
        
        # Calculate path length
        path_length = np.sum(np.abs(np.diff(prices)))
        
        # Calculate direct distance
        direct_distance = abs(prices[-1] - prices[0])
        
        if direct_distance == 0 or path_length == 0:
            return 1.5
        
        # Fractal dimension approximation
        # FD = log(path_length) / log(direct_distance)
        # Normalized to 1-2 range
        fd = 1 + (np.log(path_length) - np.log(direct_distance)) / np.log(n)
        
        # Clip to reasonable range
        fd = np.clip(fd, 1.0, 2.0)
        
        return fd
    
    fdi_values = close.rolling(window=period).apply(calculate_fdi, raw=False)
    
    fdi_values.name = f'FDI_{period}'
    return fdi_values, [fdi_values.name]


def strategy_fdi(
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
    fdi (Fractal Dimension Index) - Market Structure Strategy
    
    LOGIC: Buy when fdi drops below lower threshold (trending/persistent),
           sell when rises above upper threshold (choppy/mean-reverting).
    WHY: fdi measures market complexity. Near 1.0 = trending, near 1.5 = random,
         near 2.0 = mean-reverting. Use to select appropriate strategy.
    BEST MARKETS: All markets. Use to identify market structure.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'upper' (default 1.5),
                    'lower' (default 1.3)
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
    upper = float(parameters.get('upper', 1.5))
    lower = float(parameters.get('lower', 1.3))
    price_col = 'Close'
    indicator_col = f'FDI_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='fdi',
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
