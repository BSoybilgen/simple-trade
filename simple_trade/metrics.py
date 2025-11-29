"""
Performance metrics calculation functions for backtesting.

This module provides standalone functions for calculating various performance
metrics from backtest results, extracted from the original Backtester class.
"""
import pandas as pd
import numpy as np


def compute_benchmark_return(
    data: pd.DataFrame,
    initial_cash: float = 10000.0,
    commission_long: float = 0.001,
    price_col: str = 'Close'
) -> dict:
    """
    Computes the return from a simple buy-and-hold strategy as a benchmark.
    
    Args:
        data: DataFrame containing price data. Must have a DatetimeIndex 
              and a column specified by price_col.
        initial_cash: Starting cash balance for the benchmark.
        commission_long: Commission rate for the initial buy.
        price_col: Column name to use for price data.
        
    Returns:
        dict: Dictionary with benchmark results (final value, return, etc.)
        
    Raises:
        TypeError: If DataFrame index is not a DatetimeIndex.
        ValueError: If price column is not found in DataFrame.
        
    Example:
        >>> benchmark = compute_benchmark_return(data, initial_cash=10000)
        >>> print(f"Buy & Hold Return: {benchmark['benchmark_return_pct']:.2f}%")
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
        
    # Get first and last prices
    first_price = data[price_col].iloc[0]
    last_price = data[price_col].iloc[-1]
    
    # Calculate how many shares we could buy at the start with our initial cash, accounting for commission
    shares_bought = initial_cash / (first_price * (1 + commission_long))
    
    # Calculate the final value of our investment
    final_value = shares_bought * last_price
    
    # Calculate the return
    benchmark_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    
    # Create results dictionary
    benchmark_results = {
        "benchmark_strategy": "Buy and Hold",
        "benchmark_initial_cash": initial_cash,
        "benchmark_shares": shares_bought,
        "benchmark_buy_price": first_price,
        "benchmark_final_price": last_price,
        "benchmark_final_value": round(final_value, 2),
        "benchmark_return_pct": round(benchmark_return_pct, 2)
    }
    
    return benchmark_results


def calculate_performance_metrics(
    portfolio_df: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculates comprehensive performance metrics from a backtest.
    
    Args:
        portfolio_df: DataFrame with daily portfolio values. Must have 
                     'PortfolioValue' column and a DatetimeIndex.
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
        
    Returns:
        dict: Dictionary with performance metrics including:
            - start_date, end_date, duration_days
            - total_return_pct, annualized_return_pct
            - sharpe_ratio, sortino_ratio, calmar_ratio
            - max_drawdown_pct, avg_drawdown_pct
            - annualized_volatility_pct
            - total_commissions (if CommissionPaid column exists)
            
    Raises:
        ValueError: If portfolio_df doesn't contain 'PortfolioValue' column.
        
    Example:
        >>> metrics = calculate_performance_metrics(portfolio_df, risk_free_rate=0.02)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    if 'PortfolioValue' not in portfolio_df.columns:
        raise ValueError("portfolio_df must contain a 'PortfolioValue' column")
    
    # Start and End Date metrics
    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    duration_days = (end_date - start_date).days
    
    # Basic metrics
    initial_value = portfolio_df['PortfolioValue'].iloc[0]
    final_value = portfolio_df['PortfolioValue'].iloc[-1]
    total_return_pct = ((final_value - initial_value) / initial_value) * 100
    
    # Detect data frequency to properly annualize metrics
    time_deltas = portfolio_df.index.to_series().diff().dt.days.dropna()
    median_delta = time_deltas.median() if len(time_deltas) > 0 else 1
    
    # Determine periods per year based on data frequency
    if median_delta <= 2:  # Daily or near-daily data
        periods_per_year = 252
    elif 5 <= median_delta <= 9:  # Weekly data
        periods_per_year = 52
    elif 25 <= median_delta <= 35:  # Monthly data
        periods_per_year = 12
    else:  # Default to daily
        periods_per_year = 252
    
    # Calculate period returns (daily, weekly, etc.)
    portfolio_df = portfolio_df.copy()  # Avoid modifying original
    portfolio_df['daily_return'] = portfolio_df['PortfolioValue'].pct_change()
    
    # Annualized return and volatility
    periods_in_backtest = len(portfolio_df)
    years = periods_in_backtest / periods_per_year
    annualized_return = ((final_value / initial_value) ** (1 / years)) - 1 if years > 0 else 0
    
    # Volatility (annualized standard deviation of returns)
    period_volatility = portfolio_df['daily_return'].std()
    annualized_volatility = period_volatility * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    period_risk_free = ((1 + risk_free_rate) ** (1/periods_per_year)) - 1
    excess_return = portfolio_df['daily_return'] - period_risk_free
    if period_volatility > 1e-10 and not np.isnan(period_volatility):
        sharpe_ratio = excess_return.mean() / period_volatility * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan 
    
    # Sortino Ratio (uses downside deviation instead of total volatility)
    negative_returns = portfolio_df['daily_return'][portfolio_df['daily_return'] < 0]
    downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
        
    # Total Commissions
    total_commissions = portfolio_df['CommissionPaid'].sum() if 'CommissionPaid' in portfolio_df.columns else None
    
    # Drawdown analysis
    portfolio_df['cum_max'] = portfolio_df['PortfolioValue'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['PortfolioValue'] - portfolio_df['cum_max']) / portfolio_df['cum_max'] * 100
    max_drawdown = portfolio_df['drawdown'].min()
    avg_drawdown = portfolio_df['drawdown'][portfolio_df['drawdown'] < 0].mean() if len(portfolio_df['drawdown'][portfolio_df['drawdown'] < 0]) > 0 else 0
    
    # Drawdown duration analysis
    portfolio_df['drawdown_start'] = ~(portfolio_df['drawdown'] == 0) & (portfolio_df['drawdown'].shift(1) == 0)
    portfolio_df['drawdown_end'] = (portfolio_df['drawdown'] == 0) & ~(portfolio_df['drawdown'].shift(1) == 0)
    
    # Find all drawdown periods
    drawdown_periods = []
    in_drawdown = False
    drawdown_start = None
    
    for date, row in portfolio_df.iterrows():
        if row['drawdown_start'] and not in_drawdown:
            drawdown_start = date
            in_drawdown = True
        elif row['drawdown_end'] and in_drawdown:
            if drawdown_start is not None:
                drawdown_periods.append((drawdown_start, date, (date - drawdown_start).days))
            in_drawdown = False
            drawdown_start = None
            
    # If we're still in a drawdown at the end of the data
    if in_drawdown and drawdown_start is not None:
        drawdown_periods.append((drawdown_start, end_date, (end_date - drawdown_start).days))
    
    # Calculate drawdown duration metrics
    if drawdown_periods:
        max_drawdown_duration = max([period[2] for period in drawdown_periods])
        avg_drawdown_duration = sum([period[2] for period in drawdown_periods]) / len(drawdown_periods)
    else:
        max_drawdown_duration = 0
        avg_drawdown_duration = 0
    
    # Calmar Ratio (Annualized Return / Max Drawdown)
    calmar_ratio = annualized_return / (abs(max_drawdown) / 100) if max_drawdown != 0 else np.inf

    # Compile all metrics
    metrics = {
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
        "trading_periods": periods_in_backtest,
        "years": round(years, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "annualized_volatility_pct": round(annualized_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_drawdown_pct": round(avg_drawdown, 2),
        "max_drawdown_duration_days": max_drawdown_duration,
        "avg_drawdown_duration_days": round(avg_drawdown_duration, 2),
        "total_commissions": round(total_commissions, 2) if total_commissions is not None else None
    }
    
    return metrics


def print_results(results: dict, detailed: bool = True) -> None:
    """
    Prints the backtest results in a nicely formatted way.
    
    Args:
        results: The dictionary of backtest results.
        detailed: Whether to print detailed metrics or just basic results.
        
    Example:
        >>> print_results(results, detailed=True)
    """
    print("\n" + "="*60)
    print(f"✨ {results.get('strategy', 'Backtest Results')} ✨".center(60))
    print("="*60)
    
    # Time period information
    if 'start_date' in results and 'end_date' in results:
        print("\n🗓️ BACKTEST PERIOD:")
        start_date = results['start_date'].strftime('%Y-%m-%d') if hasattr(results['start_date'], 'strftime') else results['start_date']
        end_date = results['end_date'].strftime('%Y-%m-%d') if hasattr(results['end_date'], 'strftime') else results['end_date']
        print(f"  • Period: {start_date} to {end_date}")
        
        if 'duration_days' in results:
            print(f"  • Duration: {results['duration_days']} days")
        if 'trading_periods' in results:
            print(f"  • Trading Periods: {results['trading_periods']}")
    
    # Basic metrics section
    print("\n📊 BASIC METRICS:")
    print(f"  • Initial Investment: ${results.get('initial_cash', 0):,.2f}")
    print(f"  • Final Portfolio Value: ${results.get('final_value', 0):,.2f}")
    print(f"  • Total Return: {results.get('total_return_pct', 0):,.2f}%")
    if 'annualized_return_pct' in results:
        print(f"  • Annualized Return: {results['annualized_return_pct']:,.2f}%")
    print(f"  • Number of Trades: {results.get('num_trades', 0)}")
    if 'total_commissions' in results and results['total_commissions'] is not None:
        print(f"  • Total Commissions: ${results['total_commissions']:,.2f}")
    
    # Benchmark comparison
    if 'benchmark_return_pct' in results:
        print("\n📈 BENCHMARK COMPARISON:")
        print(f"  • Benchmark Return: {results['benchmark_return_pct']:,.2f}%")
        print(f"  • Benchmark Final Value: ${results.get('benchmark_final_value', 0):,.2f}")
        outperf = results.get('total_return_pct', 0) - results['benchmark_return_pct']
        outperf_sign = "+" if outperf >= 0 else ""
        print(f"  • Strategy vs Benchmark: {outperf_sign}{outperf:,.2f}%")
    
    # Only print detailed metrics if requested
    if detailed:
        has_risk_metrics = any(metric in results for metric in 
                           ['sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 'annualized_volatility_pct'])
        
        if has_risk_metrics:
            print("\n📉 RISK METRICS:")
            if 'sharpe_ratio' in results:
                print(f"  • Sharpe Ratio: {results['sharpe_ratio']:,.3f}")
            if 'sortino_ratio' in results:
                print(f"  • Sortino Ratio: {results['sortino_ratio']:,.3f}")
            if 'max_drawdown_pct' in results:
                print(f"  • Maximum Drawdown: {results['max_drawdown_pct']:,.2f}%")
            if 'avg_drawdown_pct' in results:
                print(f"  • Average Drawdown: {results['avg_drawdown_pct']:,.2f}%")
            if 'max_drawdown_duration_days' in results:
                print(f"  • Max Drawdown Duration: {results['max_drawdown_duration_days']} days")
            if 'avg_drawdown_duration_days' in results:
                print(f"  • Avg Drawdown Duration: {results['avg_drawdown_duration_days']} days")
            if 'annualized_volatility_pct' in results:
                print(f"  • Annualized Volatility: {results['annualized_volatility_pct']:,.2f}%")
    
    print("\n" + "="*60)


def count_trades(portfolio_df: pd.DataFrame) -> int:
    """
    Count the number of trades from a portfolio DataFrame.
    
    Args:
        portfolio_df: DataFrame with 'Action' column containing trade actions.
        
    Returns:
        int: Total number of trades executed.
    """
    if 'Action' not in portfolio_df.columns:
        return 0
    
    trade_actions = ['BUY', 'SELL', 'SHORT', 'COVER', 'COVER AND BUY', 'SELL AND SHORT']
    return len(portfolio_df[portfolio_df['Action'].isin(trade_actions)])
