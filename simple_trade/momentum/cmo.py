import pandas as pd


def cmo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Chande Momentum Oscillator (cmo), developed by Tushar Chande.
    It compares the sum of recent gains to the sum of recent losses to determine momentum.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the cmo series and a list of column names.

    The Chande Momentum Oscillator is calculated as follows:

    1. Calculate Price Differences:
       Diff = Close - Close(prev)

    2. Separate Gains and Losses:
       Gain = Diff if Diff > 0 else 0
       Loss = Abs(Diff) if Diff < 0 else 0

    3. Sum Gains and Losses over the Window:
       Sum Gains = Sum(Gain, window)
       Sum Losses = Sum(Loss, window)

    4. Calculate cmo:
       cmo = 100 * (Sum Gains - Sum Losses) / (Sum Gains + Sum Losses)

    Interpretation:
    - Range: The oscillator fluctuates between -100 and +100.
    - Overbought: Values above +50 typically indicate overbought conditions.
    - Oversold: Values below -50 typically indicate oversold conditions.
    - Trend Strength: High absolute values indicate strong trends.

    Use Cases:
    - Overbought/Oversold Levels: Identifying potential reversal points when cmo reaches extremes.
    - Trend Strength: Measuring the strength of the trend (higher absolute value = stronger trend).
    - Crosses: Crossing the zero line can be used as a signal (Bullish > 0, Bearish < 0).
    - Divergence: Divergence between price and cmo can signal potential reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    delta = series.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta.where(delta < 0, 0.0))

    sum_gains = gains.rolling(window=window, min_periods=window).sum()
    sum_losses = losses.rolling(window=window, min_periods=window).sum()
    denominator = sum_gains + sum_losses

    cmo_values = 100 * (sum_gains - sum_losses) / denominator.where(denominator != 0)
    cmo_values.name = f'CMO_{window}'

    columns_list = [cmo_values.name]
    return cmo_values, columns_list


def strategy_cmo(
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
    cmo (Chande Momentum Oscillator) - Mean Reversion Strategy
    
    LOGIC: Buy when cmo drops below -50 (oversold), sell when rises above +50 (overbought).
    WHY: cmo measures momentum using both up and down days equally weighted. Range -100 to +100.
         Extreme readings suggest exhaustion. More responsive than RSI to price changes.
    BEST MARKETS: Range-bound markets and mean-reverting assets. Stocks in consolidation,
                  forex pairs, and ETFs. Less effective in strong trends.
    TIMEFRAME: Daily or 4-hour charts. Adjust thresholds based on asset volatility.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 50), 'lower' (default -50)
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
    upper = int(parameters.get('upper', 50))
    lower = int(parameters.get('lower', -50))
    
    indicator_params = {"window": window}
    indicator_col = f'CMO_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='cmo',
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
