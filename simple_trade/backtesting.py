# Base Backtester implementation
import pandas as pd
import numpy as np

class Backtester:
    """
    Handles backtesting of trading strategies based on indicator signals.
    This is the base class with common functionality. Strategy-specific methods are in subclasses.
    """
    def __init__(self, initial_cash: float = 10000.0, commission_long: float = 0.001, 
                 commission_short: float = 0.001, short_borrow_fee_inc_rate: float = 0.0, 
                 long_borrow_fee_inc_rate: float = 0.0):
        """
        Initializes the Backtester.

        Args:
            initial_cash (float): Starting cash balance for the backtest.
            commission_long (float): Commission rate for long trades (e.g., 0.001 for 0.1%).
            commission_short (float): Commission rate for short trades (e.g., 0.001 for 0.1%).
            short_borrow_fee_inc_rate (float): Time-based fee rate for holding short positions, applied to the market value of the short position (default: 0.0).
            long_borrow_fee_inc_rate (float): Time-based fee rate for holding long positions. Typically only used for ETFs or leveraged positions (default: 0.0).
        """
        self.initial_cash = initial_cash
        self.commission_long = commission_long
        self.commission_short = commission_short
        self.short_borrow_fee_inc_rate = short_borrow_fee_inc_rate
        self.long_borrow_fee_inc_rate = long_borrow_fee_inc_rate
        
    def compute_benchmark_return(self, data: pd.DataFrame, price_col: str = 'Close') -> dict:
        """
        Computes the return from a simple buy-and-hold strategy as a benchmark.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data. Must have a DatetimeIndex and a column specified by price_col.
            price_col (str): Column name to use for price data (default: 'Close').
            
        Returns:
            dict: Dictionary with benchmark results (final value, return, etc.)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
            
        # Get first and last prices
        first_price = data[price_col].iloc[0]
        last_price = data[price_col].iloc[-1]
        
        # Calculate how many shares we could buy at the start with our initial cash, accounting for commission
        shares_bought = self.initial_cash / (first_price * (1 + self.commission_long))
        
        # Calculate the final value of our investment
        final_value = shares_bought * last_price
        
        # Calculate the return
        benchmark_return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100
        
        # Create results dictionary
        benchmark_results = {
            "benchmark_strategy": "Buy and Hold",
            "benchmark_initial_cash": self.initial_cash,
            "benchmark_shares": shares_bought,
            "benchmark_buy_price": first_price,
            "benchmark_final_price": last_price,
            "benchmark_final_value": round(final_value, 2),
            "benchmark_return_pct": round(benchmark_return_pct, 2)
        }
        
        return benchmark_results

    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
        """
        Calculates comprehensive performance metrics from a backtest.
        
        Args:
            portfolio_df (pd.DataFrame): DataFrame with daily portfolio values, must have 'PortfolioValue' column
                                         and a DatetimeIndex.
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation (default: 0.0).
            
        Returns:
            dict: Dictionary with performance metrics.
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
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['PortfolioValue'].pct_change()
        
        # Annualized return and volatility
        days_in_backtest = len(portfolio_df)
        years = days_in_backtest / 252  # Assuming 252 trading days per year
        annualized_return = ((final_value / initial_value) ** (1 / years)) - 1
        
        # Volatility (annualized standard deviation of returns)
        daily_volatility = portfolio_df['daily_return'].std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe Ratio
        daily_risk_free = ((1 + risk_free_rate) ** (1/252)) - 1
        excess_return = portfolio_df['daily_return'] - daily_risk_free
        # Handle zero or NaN volatility case for Sharpe Ratio
        if daily_volatility > 1e-10 and not np.isnan(daily_volatility):  # Use a small threshold and check for NaN
            sharpe_ratio = excess_return.mean() / daily_volatility * np.sqrt(252)
        else:
            # Assign NaN if volatility is essentially zero or NaN (Sharpe is undefined with zero risk)
            sharpe_ratio = np.nan 
        
        # Sortino Ratio (uses downside deviation instead of total volatility)
        negative_returns = portfolio_df['daily_return'][portfolio_df['daily_return'] < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
            
        # Total Commissions
        total_commissions = portfolio_df['CommissionPaid'].sum() if 'CommissionPaid' in portfolio_df.columns else None
        
        # Drawdown analysis
        portfolio_df['cum_max'] = portfolio_df['PortfolioValue'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['PortfolioValue'] - portfolio_df['cum_max']) / portfolio_df['cum_max'] * 100
        max_drawdown = portfolio_df['drawdown'].min()
        avg_drawdown = portfolio_df['drawdown'][portfolio_df['drawdown'] < 0].mean() if len(portfolio_df['drawdown'][portfolio_df['drawdown'] < 0]) > 0 else 0
        
        # Drawdown duration analysis
        # Mark drawdown start and end points
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
                if drawdown_start is not None:  # Ensure we have a valid start date
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
            "days_in_backtest": days_in_backtest,
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
