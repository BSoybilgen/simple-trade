import pandas as pd
import numpy as np
from .backtesting import Backtester

class CrossTradeBacktester(Backtester):
    """Backtester extension for Cross Trade strategies."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = None
        self.num_trades = 0
        
    def run_cross_trade(self, data: pd.DataFrame, short_window_indicator: str, 
                         long_window_indicator: str, price_col: str = 'Close', 
                         trading_type: str = 'long', long_entry_pct_cash: float = 0.9, 
                         short_entry_pct_cash: float = 0.1, day1_position: str = 'none',
                         risk_free_rate: int = 0.0) -> tuple:
        """
        Runs a backtest for a cross trading strategy using Simple Moving Averages (SMAs).

        Trading behavior is determined by the `trading_type` argument:
        - 'long' (default): 
            Buys when the short-term SMA crosses above the long-term SMA.
            Sells (closes position) when the short-term SMA crosses below the long-term SMA.
            Only long positions are allowed.
        - 'short': 
            Goes Short when the short-term SMA crosses below the long-term SMA.
            Covers Short when the short-term SMA crosses above the long-term SMA.
            No long positions are ever entered.
        - 'mixed': 
            Allows both long and short positions with seamless transitions ('Cover and Buy', 'Sell and Short')
            
        Initial position on day 1 can be set using the `day1_position` argument:
        - 'none' (default): Start with flat position, wait for signals.
        - 'long': Start with a long position on day 1.
        - 'short': Start with a short position on day 1.
        The initial position must be compatible with the trading_type (e.g., can't use 'short' with trading_type='long').

        Assumes trading at the specified price_col value on the signal day (based on previous day's crossover).

        Args:
            data (pd.DataFrame): DataFrame containing price data. Must have a DatetimeIndex and a column specified by price_col.
            short_window_indicator (str): The column name for the short-term indicator.
            long_window_indicator (str): The column name for the long-term indicator.
            price_col (str): Column name to use for trade execution prices (default: 'Close').
            trading_type (str): Defines the trading behavior. Options: 'long', 'short', 'mixed' (default: 'long').
            long_entry_pct_cash (float): Pct of available cash to use for long entries (0.0 to 1.0, default 0.9).
            short_entry_pct_cash (float): Pct of available cash defining the value of short entries (0.0 to 1.0, default 0.1).
            day1_position (str): Specifies whether to take a position on day 1. Options: 'none', 'long', 'short' (default: 'none').
            risk_free_rate (float): Risk-free rate for Sharpe and Sortino ratios (default: 0.0).

        Returns:
            tuple: A tuple containing:
                - dict: Dictionary with backtest summary results (final value, return, trades).
                - pd.DataFrame: DataFrame tracking daily portfolio evolution (cash, position, value, signals, actions).
        """
        # Initialize portfolio log list for storing daily state
        self.portfolio_log = []
        
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

        # Explicitly check for short_window_indicator and raise the specific error
        if short_window_indicator not in data.columns:
            raise ValueError(f"Required column '{short_window_indicator}' for short window indicator is missing from the DataFrame.")

        if long_window_indicator not in data.columns:
            raise ValueError(f"Required column '{long_window_indicator}' for long window indicator is missing from the DataFrame.")
        if not (0.0 <= long_entry_pct_cash <= 1.0):
            raise ValueError("long_entry_pct_cash must be between 0.0 and 1.0")
        if not (0.0 <= short_entry_pct_cash <= 1.0):
            raise ValueError("short_entry_pct_cash must be between 0.0 and 1.0")

        valid_trading_types = ['long', 'short', 'mixed']
        if trading_type not in valid_trading_types:
            raise ValueError(f"Invalid trading_type '{trading_type}'. Must be one of {valid_trading_types}")
            
        # Validate day1_position
        valid_day1_positions = ['none', 'long', 'short']
        if day1_position not in valid_day1_positions:
            raise ValueError(f"Invalid day1_position '{day1_position}'. Must be one of {valid_day1_positions}")
        
        # Check compatibility between day1_position and trading_type
        if day1_position == 'long' and trading_type == 'short':
            raise ValueError("Cannot use day1_position='long' with trading_type='short'")
        if day1_position == 'short' and trading_type == 'long':
            raise ValueError("Cannot use day1_position='short' with trading_type='long'")

        df = data.copy() # Work on a copy

        # --- Signal Generation ---
        # Use shift(1) to base signals on the *previous* day's crossover
        prev_short = df[short_window_indicator].shift(1)
        prev_long = df[long_window_indicator].shift(1)
        prev_prev_short = df[short_window_indicator].shift(2) # Need previous-previous for crossover check
        prev_prev_long = df[long_window_indicator].shift(2)

        # Golden Cross (Buy/Cover Signal): Short crossed above Long on the *previous* day
        df['buy_signal'] = (prev_short > prev_long) & (prev_prev_short <= prev_prev_long)

        # Death Cross (Sell/Short Signal): Short crossed below Long on the *previous* day
        df['sell_signal'] = (prev_short < prev_long) & (prev_prev_short >= prev_prev_long)

        # Drop NaNs created by shifts
        df = df.dropna(how='any') # Explicitly use how='any'

        # --- Check if DataFrame is empty AFTER potential drops ---
        if df.empty:
            self.log.warning(f"DataFrame is empty after generating signals and dropping NaNs for indicators '{short_window_indicator}' and '{long_window_indicator}'. No trades executed.")
            # Return minimal results, NO performance metrics calculated
            strategy_name_early = f"Cross Trade ({short_window_indicator}/{long_window_indicator}){' [Shorts Allowed]' if trading_type in ['short', 'mixed'] else ''}{' [Day1 ' + day1_position.capitalize() + ']' if day1_position != 'none' else ''}"
            early_results = {
                "strategy": strategy_name_early,
                "short_window_indicator": short_window_indicator,
                "long_window_indicator": long_window_indicator,
                "initial_cash": self.initial_cash,
                "final_value": self.initial_cash, # No trades, value is initial cash
                "total_return_pct": 0.0,
                "num_trades": 0,
            }
            return early_results, pd.DataFrame() # Return empty DataFrame

        # --- Initialize State --- Only run if df was not empty ---
        cash = self.initial_cash
        position_size = 0 # Shares held (negative for short positions)
        position_type = 'none' # Position type ('none', 'long', 'short')
        position_cost_basis = 0 # Weighted average cost of current position
        self.num_trades = 0 # Reset trade counter for this run
        portfolio_log = [] # Initialize portfolio log locally for consistency with BandTradeBacktester
        
        # Initialize variable to track if we're on the first day
        first_day = True

        # --- Backtesting Loop: Process Each Day ---
        for idx, row in df.iterrows():
            # Today's prices and signals
            trade_price = row[price_col]
            
            # Special handling for first day if day1_position is specified
            if first_day and day1_position != 'none':
                # Override signals for day 1
                buy_signal = day1_position == 'long'
                sell_signal = day1_position == 'short'
                first_day = False  # No longer first day after this
            else:
                buy_signal = row['buy_signal']
                sell_signal = row['sell_signal']
            
            # Initialize default state
            signal_generated = "NONE"
            action_taken = "HOLD"
            commission_paid = 0.0
            short_fee = 0.0
            long_fee = 0.0
            
            # Calculate position value based on position size and type
            position_value = 0.0
            if position_type == 'long':
                position_value = position_size * trade_price
                
                # Apply daily borrow fee for long positions if applicable (e.g., leveraged ETFs)
                if self.long_borrow_fee_inc_rate > 0:
                    long_fee = position_value * self.long_borrow_fee_inc_rate
                    cash -= long_fee
            
            elif position_type == 'short':
                position_value = abs(position_size) * trade_price  # Positive value representing liability
                
                # Apply daily borrow fee for short positions if applicable
                if self.short_borrow_fee_inc_rate > 0:
                    short_fee = position_value * self.short_borrow_fee_inc_rate
                    cash -= short_fee
            
            # Calculate portfolio value (consistent with band_trade.py)
            portfolio_value = cash
            if position_type == 'long':
                portfolio_value += position_value
            elif position_type == 'short':
                # For short positions, subtract the positive position value (liability)
                portfolio_value -= position_value
            
            # --- Execute Trading Logic Based on trading_type ---
            
            if trading_type == 'long': # LONG-ONLY strategy
                if position_type == 'none' and buy_signal: # We're flat and have a buy signal
                    # Enter long position
                    signal_generated = "Buy"
                    action_taken = "BUY"
                    
                    # Calculate shares to buy (consider commission in calculation)
                    max_shares = int((cash * long_entry_pct_cash) / (trade_price * (1 + self.commission_long)))
                    
                    if max_shares > 0:
                        position_size = max_shares
                        position_type = 'long'  # Set position type explicitly
                        commission_cost = position_size * trade_price * self.commission_long
                        cash -= (position_size * trade_price + commission_cost)
                        position_cost_basis = trade_price
                        commission_paid = commission_cost
                        self.num_trades += 1
                    else:
                        action_taken = "INSUFFICIENT_CASH"
                        
                elif position_type == 'long' and sell_signal: # We have a long position and need to sell signal
                    # Close long position
                    signal_generated = "Sell"
                    action_taken = "SELL"
                    
                    # Calculate proceeds from sale (consider commission)
                    commission_cost = position_size * trade_price * self.commission_long
                    cash += (position_size * trade_price - commission_cost)
                    commission_paid = commission_cost
                    
                    position_size = 0
                    position_type = 'none'  # Reset position type when closing position
                    position_cost_basis = 0
                    self.num_trades += 1

            elif trading_type == 'short': # SHORT-ONLY strategy
                if position_type == 'none' and sell_signal: # We're flat and have a sell signal
                    # Enter short position
                    signal_generated = "Short"
                    action_taken = "SHORT"
                    
                    # Calculate shares to short (careful with cash calculation)
                    short_position_value = cash * short_entry_pct_cash
                    max_shares = int(short_position_value / (trade_price * (1 + self.commission_short)))
                    
                    if max_shares > 0:
                        position_size = -max_shares  # Negative for short
                        position_type = 'short'  # Set position type explicitly
                        commission_cost = abs(position_size) * trade_price * self.commission_short
                        # When shorting, cash INCREASES (we receive proceeds from the short sale)
                        cash += (abs(position_size) * trade_price - commission_cost)
                        position_cost_basis = trade_price
                        commission_paid = commission_cost
                        self.num_trades += 1
                    else:
                        action_taken = "INSUFFICIENT_CASH"
                        
                elif position_type == 'short' and buy_signal: # We're short and have a buy signal
                    # Cover short position
                    signal_generated = "Cover"
                    action_taken = "COVER"
                    
                    # Calculate cost to buy back shares (consider commission)
                    commission_cost = abs(position_size) * trade_price * self.commission_short
                    # When covering, cash DECREASES (we pay to buy back the shares)
                    cash -= (abs(position_size) * trade_price + commission_cost)
                    commission_paid = commission_cost
                    
                    position_size = 0
                    position_type = 'none'
                    position_cost_basis = 0
                    self.num_trades += 1

            else: # MIXED strategy (both long and short with possible direct transitions)
                if position_type == 'none': # Flat position
                    if buy_signal: # Enter long
                        signal_generated = "Buy"
                        action_taken = "BUY"
                        
                        max_shares = int((cash * long_entry_pct_cash) / (trade_price * (1 + self.commission_long)))
                        
                        if max_shares > 0:
                            position_size = max_shares
                            position_type = 'long'  # Set position type explicitly
                            commission_cost = position_size * trade_price * self.commission_long
                            cash -= (position_size * trade_price + commission_cost)
                            position_cost_basis = trade_price
                            commission_paid = commission_cost
                            self.num_trades += 1
                        else:
                            action_taken = "INSUFFICIENT_CASH"
                            
                    elif sell_signal: # Enter short
                        signal_generated = "Short"
                        action_taken = "SHORT"
                        
                        short_position_value = cash * short_entry_pct_cash
                        max_shares = int(short_position_value / (trade_price * (1 + self.commission_short)))
                        
                        if max_shares > 0:
                            position_size = -max_shares
                            position_type = 'short'  # Set position type explicitly
                            commission_cost = abs(position_size) * trade_price * self.commission_short
                            cash += (abs(position_size) * trade_price - commission_cost)
                            position_cost_basis = trade_price
                            commission_paid = commission_cost
                            self.num_trades += 1
                        else:
                            action_taken = "INSUFFICIENT_CASH"
                
                elif position_type == 'long': # Long position
                    if sell_signal: # Have sell signal while long
                        signal_generated = "Sell"
                        
                        if buy_signal:  # Both buy AND sell signals - conflicting
                            action_taken = "HOLD_CONFLICTING_SIGNAL"
                        else:
                            # Determine if we should just sell or "Sell and Short" based on logic
                            # We have an explicit sell signal, so we'll flip to short
                            signal_generated = "Sell and Short"
                            action_taken = "SELL AND SHORT"
                            
                            # First close the long position
                            commission_cost = position_size * trade_price * self.commission_long
                            cash += (position_size * trade_price - commission_cost)
                            commission_paid = commission_cost
                            position_size = 0
                            position_type = 'none'  # Reset position type after closing long position
                            
                            # Then enter short position using available cash
                            short_position_value = cash * short_entry_pct_cash
                            max_shares = int(short_position_value / (trade_price * (1 + self.commission_short)))
                            
                            if max_shares > 0:
                                position_size = -max_shares
                                position_type = 'short'  # Set position type when entering short
                                commission_cost = abs(position_size) * trade_price * self.commission_short
                                cash += (abs(position_size) * trade_price - commission_cost)
                                commission_paid += commission_cost  # Add to existing commission
                                position_cost_basis = trade_price
                                self.num_trades += 2  # Count as two trades (sell and short)
                            else:
                                action_taken = "SELL" # Just sell if can't short
                                self.num_trades += 1
                
                elif position_type == 'short': # Short position
                    if buy_signal: # Have buy signal while short
                        signal_generated = "Cover"
                        
                        if sell_signal:  # Both buy AND sell signals - conflicting
                            action_taken = "HOLD_CONFLICTING_SIGNAL"
                        else:
                            # Determine if we should just cover or "Cover and Buy" based on logic
                            # We have an explicit buy signal, so we'll flip to long
                            signal_generated = "Cover and Buy"
                            action_taken = "COVER AND BUY"
                            
                            # First cover the short position
                            commission_cost = abs(position_size) * trade_price * self.commission_short
                            cash -= (abs(position_size) * trade_price + commission_cost)
                            commission_paid = commission_cost
                            position_size = 0
                            position_type = 'none'  # Reset position type after covering short
                            
                            # Then enter long position using available cash
                            max_shares = int((cash * long_entry_pct_cash) / (trade_price * (1 + self.commission_long)))
                            
                            if max_shares > 0:
                                position_size = max_shares
                                position_type = 'long'  # Set position type when entering long
                                commission_cost = position_size * trade_price * self.commission_long
                                cash -= (position_size * trade_price + commission_cost)
                                commission_paid += commission_cost  # Add to existing commission
                                position_cost_basis = trade_price
                                self.num_trades += 2  # Count as two trades (cover and buy)
                            else:
                                action_taken = "COVER" # Just cover if can't buy
                                self.num_trades += 1
                
            # --- Log Daily State (using self.portfolio_log) ---
            # Convert signal_generated to boolean buy/sell signals for consistency with band_trade.py
            buy_signal = False
            sell_signal = False
            if signal_generated in ['Buy', 'Cover and Buy']:
                buy_signal = True
            elif signal_generated in ['Sell', 'Sell and Short']:
                sell_signal = True
            elif signal_generated == 'Cover':
                buy_signal = True  # Cover is triggered by buy signals
            elif signal_generated == 'Short':
                sell_signal = True  # Short is triggered by sell signals
                
            log_entry = {
                'Date': idx,
                'Price': trade_price,  # Renamed from TradePrice for consistency
                'Close': trade_price,  # Added for compatibility with compute_benchmark_return
                'Cash': cash,
                'PositionSize': position_size,
                'PositionValue': position_value,  # Use calculated position value for consistency
                'PositionType': position_type,  # Added for consistency
                'PortfolioValue': portfolio_value,
                'CommissionPaid': commission_paid,
                'ShortFee': short_fee,
                'LongFee': long_fee,
                'BuySignal': buy_signal,  # Added for consistency
                'SellSignal': sell_signal,  # Added for consistency
                'Action': action_taken,
                'PositionCostBasis': position_cost_basis  # Keep this unique field
            }
            # --------------------------------------------------
            portfolio_log.append(log_entry)

        # --- Final Calculations and DataFrame Creation ---
        # Create portfolio DataFrame from log
        portfolio_df = pd.DataFrame(portfolio_log)
        
        if not portfolio_log: # Check if the log itself is empty first
             portfolio_df = pd.DataFrame() # Ensure df is truly empty
        else:
            portfolio_df.set_index('Date', inplace=True)
            # Ensure index is a DatetimeIndex if conversion happened
            if not isinstance(portfolio_df.index, pd.DatetimeIndex):
                 portfolio_df.index = pd.to_datetime(portfolio_df.index)

        # --- Calculate Final Value and Basic Results ---
        final_value = portfolio_df['PortfolioValue'].iloc[-1] if not portfolio_df.empty else self.initial_cash
        total_return_pct = ((final_value / self.initial_cash) - 1) * 100 if self.initial_cash else 0

        strategy_name = f"Cross Trade ({short_window_indicator}/{long_window_indicator}){' [Shorts Allowed]' if trading_type in ['short', 'mixed'] else ''}{' [Day1 ' + day1_position.capitalize() + ']' if day1_position != 'none' else ''}"
        results = {
            "strategy": strategy_name,
            "short_window_indicator": short_window_indicator,
            "long_window_indicator": long_window_indicator,
            "initial_cash": self.initial_cash,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "num_trades": self.num_trades,
        }

        # --- Calculate Benchmark and Performance Metrics using Base Class Methods ---
        # The earlier 'if df.empty:' check ensures we don't reach here with an effectively empty trading period.
        # The portfolio_df created from the log will have at least the initial state row.
        benchmark_results = self.compute_benchmark_return(data, price_col=price_col) # Use original data for benchmark
        performance_metrics = self.calculate_performance_metrics(portfolio_df.copy(), risk_free_rate) # Pass a copy to avoid inplace modification issues

        # Update the main results dictionary
        results.update(benchmark_results)
        results.update(performance_metrics)

        # Return the combined results and the portfolio DataFrame
        # Note: We passed a copy to metrics, so return the original portfolio_df
        return results, portfolio_df
        
    def print_results(self, results: dict, detailed: bool = True):
        """
        Prints the backtest results in a nicely formatted way.
        
        Args:
            results (dict): The dictionary of backtest results.
            detailed (bool): Whether to print detailed metrics or just basic results.
        """
        print("\n" + "="*60)
        print(f"✨ {results['strategy']} ✨".center(60))
        print("="*60)
        
        # Time period information
        if 'start_date' in results and 'end_date' in results:
            print("\n🗓️ BACKTEST PERIOD:")
            start_date = results['start_date'].strftime('%Y-%m-%d') if hasattr(results['start_date'], 'strftime') else results['start_date']
            end_date = results['end_date'].strftime('%Y-%m-%d') if hasattr(results['end_date'], 'strftime') else results['end_date']
            print(f"  • Period: {start_date} to {end_date}")
            
            if 'duration_days' in results:
                print(f"  • Duration: {results['duration_days']} days")
            if 'days_in_backtest' in results:
                print(f"  • Trading Days: {results['days_in_backtest']}")
        
        # Basic metrics section
        print("\n📊 BASIC METRICS:")
        print(f"  • Initial Investment: ${results['initial_cash']:,.2f}")
        print(f"  • Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"  • Total Return: {results['total_return_pct']:,.2f}%")
        if 'annualized_return_pct' in results:
            print(f"  • Annualized Return: {results['annualized_return_pct']:,.2f}%")
        print(f"  • Number of Trades: {results['num_trades']}")
        if 'total_commissions' in results and results['total_commissions'] is not None:
            print(f"  • Total Commissions: ${results['total_commissions']:,.2f}")
        
        # Benchmark comparison
        if 'benchmark_return_pct' in results:
            print("\n📈 BENCHMARK COMPARISON:")
            print(f"  • Benchmark Return: {results['benchmark_return_pct']:,.2f}%")
            print(f"  • Benchmark Final Value: ${results['benchmark_final_value']:,.2f}")
            outperf = results['total_return_pct'] - results['benchmark_return_pct']
            outperf_sign = "+" if outperf >= 0 else ""
            print(f"  • Strategy vs Benchmark: {outperf_sign}{outperf:,.2f}%")
        
        # Only print detailed metrics if requested
        if detailed:
            # Risk metrics
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
