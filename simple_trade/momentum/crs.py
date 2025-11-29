import numpy as np
import pandas as pd


def crs(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Connors RSI (CRSI), a composite indicator designed by Larry Connors 
    to better identify short-term overbought and oversold conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - rsi_window (int): Window for the close-price RSI component. Default is 3.
            - streak_window (int): Window for the streak RSI component. Default is 2.
            - rank_window (int): Lookback period for the percent rank component. Default is 100.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Connors RSI series and a list of column names.

    The Connors RSI is calculated as the average of three components:

    1. Calculate the RSI of Closing Prices:
       RSI_Price = RSI(Close, rsi_window)

    2. Calculate the RSI of the Streak:
       - Determine the Streak (consecutive days up or down).
       - RSI_Streak = RSI(Streak, streak_window)

    3. Calculate the Percent Rank of Price Change:
       - Calculate one-day price change (Percent Change).
       - Rank = Percent Rank of today's change over rank_window.

    4. Calculate CRSI:
       CRSI = (RSI_Price + RSI_Streak + Rank) / 3

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 90 (or 95) indicate potential short-term exhaustion/overbought.
    - Oversold: Values below 10 (or 5) indicate potential short-term oversold conditions.

    Use Cases:
    - Mean Reversion: Identifying short-term pullback opportunities in a broader trend.
    - Exit Signals: Exiting positions when CRSI reaches extreme levels.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    rsi_window = int(parameters.get('rsi_window', 3))
    streak_window = int(parameters.get('streak_window', 2))
    rank_window = int(parameters.get('rank_window', 100))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        # Handle edge cases: when avg_loss is 0, RSI should be 100 (all gains)
        # When avg_gain is 0, RSI should be 0 (all losses)
        # When both are 0, RSI should be 50 (neutral)
        rsi_values = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            ag = avg_gain.iloc[i]
            al = avg_loss.iloc[i]
            if pd.isna(ag) or pd.isna(al):
                rsi_values.iloc[i] = np.nan
            elif al == 0 and ag == 0:
                rsi_values.iloc[i] = 50.0  # Neutral when no movement
            elif al == 0:
                rsi_values.iloc[i] = 100.0  # All gains
            else:
                rs = ag / al
                rsi_values.iloc[i] = 100 - (100 / (1 + rs))
        return rsi_values

    # Component 1: RSI of closing prices
    price_rsi = _rsi(close, rsi_window)

    # Component 2: RSI of streak length
    delta = close.diff()
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak.iloc[i] = streak.iloc[i - 1] + 1 if streak.iloc[i - 1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak.iloc[i] = streak.iloc[i - 1] - 1 if streak.iloc[i - 1] < 0 else -1
        else:
            streak.iloc[i] = 0
    streak_rsi = _rsi(streak, streak_window)

    # Component 3: Percent rank of price change
    price_change = close.diff()

    def _percent_rank(window_values):
        valid = window_values[~pd.isna(window_values)]
        if len(valid) < len(window_values) or len(valid) <= 1:
            return np.nan
        last = valid.iloc[-1]
        less_count = (valid.iloc[:-1] < last).sum()
        equal_count = (valid.iloc[:-1] == last).sum()
        denom = len(valid) - 1
        if denom <= 0:
            return np.nan
        rank = (less_count + 0.5 * equal_count) / denom
        return rank * 100

    percent_rank = price_change.rolling(window=rank_window, min_periods=rank_window).apply(
        lambda x: _percent_rank(pd.Series(x)), raw=False
    )

    percent_rank = percent_rank.astype(float)

    crsi_values = (price_rsi + streak_rsi + percent_rank) / 3
    crsi_values.name = f'CRSI_{rsi_window}_{streak_window}_{rank_window}'

    columns_list = [crsi_values.name]
    return crsi_values, columns_list


def strategy_crs(
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
    CRS (Connors RSI) - Mean Reversion Strategy
    
    LOGIC: Buy when CRSI drops below 10 (extreme oversold), sell when rises above 90.
    WHY: CRSI combines 3 components (RSI, streak RSI, percent rank) for short-term
         mean reversion. Designed by Larry Connors for identifying pullback opportunities.
    BEST MARKETS: Liquid stocks and ETFs in uptrends. Best for buying dips in bull markets.
                  SPY, QQQ, and large-cap stocks. Avoid in bear markets or downtrends.
    TIMEFRAME: Daily charts. Designed for short-term trades (2-5 day holding periods).
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'rsi_window' (default 3), 'streak_window' (default 2),
                   'rank_window' (default 100), 'upper' (default 90), 'lower' (default 10)
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
    
    rsi_window = int(parameters.get('rsi_window', 3))
    streak_window = int(parameters.get('streak_window', 2))
    rank_window = int(parameters.get('rank_window', 100))
    upper = int(parameters.get('upper', 90))
    lower = int(parameters.get('lower', 10))
    
    indicator_params = {
        "rsi_window": rsi_window,
        "streak_window": streak_window,
        "rank_window": rank_window
    }
    indicator_col = f'CRSI_{rsi_window}_{streak_window}_{rank_window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='crs',
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
