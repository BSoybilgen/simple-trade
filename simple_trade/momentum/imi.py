import pandas as pd


def imi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Intraday Momentum Index (IMI), a technical indicator developed by Tushar Chande 
    that combines candlestick analysis with the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the IMI series and a list of column names.

    The Intraday Momentum Index is calculated as follows:

    1. Calculate Intraday Gains and Losses:
       If Close > Open: Gain = Close - Open, Loss = 0
       If Close < Open: Loss = Open - Close, Gain = 0

    2. Sum Gains and Losses over the Window:
       Sum Gains = Sum(Gain, window)
       Sum Losses = Sum(Loss, window)

    3. Calculate IMI:
       IMI = (Sum Gains / (Sum Gains + Sum Losses)) * 100

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.
    - Momentum: High IMI suggests strong buying pressure within the day (white/green candles dominate).
      Low IMI suggests strong selling pressure (black/red candles dominate).

    Use Cases:
    - Reversal Signals: Identifying overbought/oversold levels for potential reversals.
    - Confirmation: Using IMI to confirm support/resistance levels.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    open_col = columns.get('open_col', 'Open')
    close_col = columns.get('close_col', 'Close')

    open_vals = df[open_col]
    close_vals = df[close_col]

    # Calculate gains (up candles) and losses (down candles)
    # Gain = Close - Open (if Close > Open)
    # Loss = Open - Close (if Close < Open)

    diff = close_vals - open_vals
    gains = diff.where(diff > 0, 0)
    losses = -diff.where(diff < 0, 0)

    sum_gains = gains.rolling(window=window).sum()
    sum_losses = losses.rolling(window=window).sum()

    imi_val = (sum_gains / (sum_gains + sum_losses)) * 100
    imi_val.name = f'IMI_{window}'

    return imi_val, [imi_val.name]


def strategy_imi(
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
    IMI (Intraday Momentum Index) - Mean Reversion Strategy
    
    LOGIC: Buy when IMI drops below lower threshold (oversold), sell when above upper.
    WHY: IMI combines candlestick analysis with RSI logic. Measures buying vs selling
         pressure within each bar. High IMI = strong intraday buyers, low = sellers.
    BEST MARKETS: Stocks and ETFs with significant intraday range. Range-bound markets.
                  Good for identifying exhaustion in short-term moves.
    TIMEFRAME: Daily charts. 14-period is standard. Works well for swing trading reversals.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 70), 'lower' (default 30)
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
    
    window = int(parameters.get('window', 14))
    upper = int(parameters.get('upper', 70))
    lower = int(parameters.get('lower', 30))
    
    indicator_params = {"window": window}
    indicator_col = f'IMI_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='imi',
        parameters=indicator_params,
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
