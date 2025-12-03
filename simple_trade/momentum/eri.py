import pandas as pd


def eri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Elder Ray Index (eri), a technical indicator developed by Dr. Alexander Elder.
    It measures the amount of buying and selling pressure in the market.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The period for the EMA baseline. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the eri DataFrame (with Bull and Bear power) and a list of column names.

    The eri consists of three components (calculated as follows):

    1. Calculate the Exponential Moving Average (EMA):
       EMA = EMA(Close, window) (Often a 13-period EMA)

    2. Calculate Bull Power:
       Bull Power = High - EMA

    3. Calculate Bear Power:
       Bear Power = Low - EMA

    Interpretation:
    - Bull Power: Measures the ability of buyers to push prices above the average consensus of value (EMA).
      Positive values indicate strength.
    - Bear Power: Measures the ability of sellers to push prices below the average consensus of value (EMA).
      Negative values indicate weakness.
    - EMA Slope: Indicates the direction of the main trend.

    Use Cases:
    - Trend Following: Buy when the trend is up (EMA rising) and Bear Power is negative but rising.
      Sell when the trend is down (EMA falling) and Bull Power is positive but falling.
    - Divergence: Divergence between prices and Bull/Bear Power can signal reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 13))
    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    ema = close.ewm(span=window, adjust=False, min_periods=window).mean()

    bull_col = f'ERI_BULL_{window}'
    bear_col = f'ERI_BEAR_{window}'
    result = pd.DataFrame({
        bull_col: high - ema,
        bear_col: low - ema
    })

    columns_list = [bull_col, bear_col]
    return result, columns_list


def strategy_eri(
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
    eri (Elder Ray Index) - Bull/Bear Power Strategy
    
    LOGIC: Buy when Bull Power is positive and rising, sell when Bear Power is negative and falling.
    WHY: eri measures buying/selling pressure relative to EMA. Positive Bull Power
         = buyers pushing above average, negative Bear Power = sellers pushing below prices below consensus value.
    BEST MARKETS: Trending stocks and indices with clear institutional participation.
                  Works well when combined with EMA trend filter.
    TIMEFRAME: Daily charts. 13-period EMA is standard (Elder's recommendation).
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 13)
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
    
    window = int(parameters.get('window', 13))
    
    indicator_params = {"window": window}
    short_window_indicator = f'ERI_BEAR_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='eri',
        parameters=indicator_params,
        figure=False
    )
    
    # Create zero line for crossover strategy
    data['zero_line'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [f'ERI_BULL_{window}', f'ERI_BEAR_{window}', 'zero_line']
    
    return results, portfolio, indicator_cols_to_plot, data
