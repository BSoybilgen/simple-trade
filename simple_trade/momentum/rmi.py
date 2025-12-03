import numpy as np
import pandas as pd


def rmi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Momentum Index (rmi), a variation of the Relative Strength Index (RSI).
    Instead of using day-to-day price changes, RMI uses price changes over a specified momentum period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The smoothing window for average gains/losses. Default is 20.
            - momentum_period (int): The lookback period for momentum difference. Default is 5.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RMI series and a list of column names.

    The Relative Momentum Index is calculated as follows:

    1. Calculate Momentum:
       Momentum = Close - Close(shifted by momentum_period)

    2. Separate Gains and Losses:
       Gain = Momentum if Momentum > 0 else 0
       Loss = Abs(Momentum) if Momentum < 0 else 0

    3. Calculate Average Gain and Loss:
       Avg Gain = SMA(Gain, window)
       Avg Loss = SMA(Loss, window)

    4. Calculate RMI:
       RS = Avg Gain / Avg Loss
       RMI = 100 - (100 / (1 + RS))

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.
    - Divergence: Divergence between price and RMI can signal reversals.

    Use Cases:
    - Cycle Analysis: RMI is often smoother than RSI and can better highlight cyclical turns.
    - Trend Confirmation: Confirming the direction of the trend.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 20))
    momentum_period = int(parameters.get('momentum_period', 5))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    momentum = series - series.shift(momentum_period)

    gains = momentum.where(momentum > 0, 0.0)
    losses = (-momentum.where(momentum < 0, 0.0))

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace({0: np.nan})
    rmi_values = 100 - (100 / (1 + rs))
    rmi_values = rmi_values.astype(float)
    rmi_values.name = f'RMI_{window}_{momentum_period}'

    columns_list = [rmi_values.name]
    return rmi_values, columns_list


def strategy_rmi(
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
    rmi (Relative Momentum Index) - Mean Reversion Strategy
    
    LOGIC: Buy when rmi drops below lower threshold (oversold), sell when above upper.
    WHY: rmi is RSI with momentum lookback instead of 1-day changes. Smoother than RSI,
         better at highlighting cyclical turns. Momentum period adds flexibility.
    BEST MARKETS: Range-bound markets and cyclical assets. Stocks, forex, commodities.
                  Smoother signals reduce whipsaws compared to standard RSI.
    TIMEFRAME: Daily charts. 20-period with 5-day momentum is common.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'momentum_period' (default 5),
                   'upper' (default 70), 'lower' (default 30)
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
    momentum_period = int(parameters.get('momentum_period', 5))
    upper = int(parameters.get('upper', 70))
    lower = int(parameters.get('lower', 30))
    
    indicator_params = {"window": window, "momentum_period": momentum_period}
    indicator_col = f'RMI_{window}_{momentum_period}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='rmi',
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
