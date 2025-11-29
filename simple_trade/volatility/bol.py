import pandas as pd
import numpy as np


def bol(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Bollinger Bands of a series.
    Bollinger Bands are a type of statistical chart illustrating the relative high and low prices
    of a security in relation to its average price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for calculating the moving average and standard deviation. Default is 20.
            - num_std (int): The number of standard deviations to use for the upper and lower bands. Default is 2.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Bollinger Bands DataFrame and a list of column names.

    Calculation Steps:
    1. Middle Band:
       SMA of the price over the window.
    2. Standard Deviation:
       Calculate standard deviation of price over the window.
    3. Upper Band:
       Middle Band + (num_std * Standard Deviation)
    4. Lower Band:
       Middle Band - (num_std * Standard Deviation)

    Interpretation:
    - Price near Upper Band: Potential overbought condition.
    - Price near Lower Band: Potential oversold condition.
    - Squeeze: Bands contracting indicates low volatility and potential breakout.
    - Expansion: Bands widening indicates increasing volatility.

    Use Cases:
    - Identifying overbought/oversold conditions.
    - Measuring volatility (Bandwidth).
    - Generating buy/sell signals on breakouts or reversals at bands.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    num_std = float(parameters.get('num_std', 2))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]

    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    # Return DataFrame for multi-output indicators
    df_bb = pd.DataFrame({
        f'BB_Middle_{window}': sma,
        f'BB_Upper_{window}_{num_std}': upper_band,
        f'BB_Lower_{window}_{num_std}': lower_band
    })
    # Ensure index is passed explicitly, just in case
    df_bb.index = series.index
    columns_list = list(df_bb.columns)
    return df_bb, columns_list


def strategy_bol(
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
    BOL (Bollinger Bands) - Mean Reversion Strategy
    
    LOGIC: Buy when price touches lower band (oversold),
           sell when price touches upper band (overbought).
    WHY: Bollinger Bands show price relative to volatility. Price at lower band
         suggests oversold, at upper band suggests overbought.
    BEST MARKETS: Range-bound markets. Stocks, forex. Good for mean reversion.
                  Less effective in strong trends.
    TIMEFRAME: Daily charts. 20-period with 2 std dev is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'num_std' (default 2.0)
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
    num_std = float(parameters.get('num_std', 2.0))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='bol',
        parameters={"window": window, "num_std": num_std},
        figure=False
    )
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col='Close',
        upper_band_col=f'BB_Upper_{window}_{num_std}',
        lower_band_col=f'BB_Lower_{window}_{num_std}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', f'BB_Upper_{window}_{num_std}', 
                              f'BB_Middle_{window}', f'BB_Lower_{window}_{num_std}']
    
    return results, portfolio, indicator_cols_to_plot, data