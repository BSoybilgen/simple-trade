import pandas as pd
import numpy as np


def uli(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ulcer Index (uli), a volatility indicator that measures downside risk
    by focusing on the depth and duration of price drawdowns from recent peaks, rather
    than treating upside and downside volatility equally.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Ulcer Index series and a list of column names.

    Calculation Steps:
    1. Calculate Percentage Drawdown:
       For each period, find the highest close over the lookback window.
       Drawdown = 100 * (Close - Highest Close) / Highest Close

    2. Calculate Squared Drawdowns:
       Squared Drawdown = (Drawdown)^2

    3. Calculate Mean Squared Drawdown:
       Mean = Sum(Squared Drawdowns) / period

    4. Calculate Ulcer Index:
       ULI = sqrt(Mean)

    Interpretation:
    - Higher ULI values indicate greater downside risk and deeper/longer drawdowns.
    - Lower ULI values indicate stability or upward trends.
    - Unlike standard deviation, ULI does not penalize upside volatility.

    Use Cases:
    - Downside risk measurement.
    - Portfolio optimization (e.g., Ulcer Performance Index).
    - Drawdown monitoring and risk management.
    - Comparison of strategies with similar returns but different risk profiles.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate the highest close over the rolling period
    highest_close = close.rolling(window=period).max()
    
    # Calculate percentage drawdown from the highest close
    # Drawdown % = 100 * (Close - Highest Close) / Highest Close
    drawdown_pct = 100 * (close - highest_close) / highest_close
    
    # Square each drawdown percentage
    squared_drawdown = drawdown_pct ** 2
    
    # Calculate the mean of squared drawdowns over the period
    mean_squared_drawdown = squared_drawdown.rolling(window=period).mean()
    
    # Take the square root to get the Ulcer Index
    ulcer_index = np.sqrt(mean_squared_drawdown)
    
    ulcer_index.name = f'ULI_{period}'
    columns_list = [ulcer_index.name]
    return ulcer_index, columns_list


def strategy_uli(
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
    uli (Ulcer Index) - Downside Risk Strategy
    
    LOGIC: Buy when uli drops below lower threshold (low downside risk),
           sell when rises above upper threshold (high downside risk).
    WHY: uli measures downside risk by focusing on drawdowns.
         Unlike std dev, it doesn't penalize upside volatility.
    BEST MARKETS: All markets. Good for downside risk measurement.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14), 'upper' (default 10),
                    'lower' (default 3)
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
    
    period = int(parameters.get('period', 14))
    upper = float(parameters.get('upper', 10))
    lower = float(parameters.get('lower', 3))
    price_col = 'Close'
    indicator_col = f'ULI_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='uli',
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
